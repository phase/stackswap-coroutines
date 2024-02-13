#![feature(strict_provenance)]

use setjmp::*;
use std::cell::RefCell;
use std::ffi::c_void;
use std::mem::MaybeUninit;
use std::{ptr, thread_local};

thread_local! {
    static CURRENT: RefCell<Option<*mut Stack>> = RefCell::new(None);
}

pub struct Dispatcher {
    jmpbuf: *mut jmp_buf,
    allocator: Box<dyn StackBufferAllocator>,
    stacks: Vec<Stack>,
    current_stack: usize,
}

impl Dispatcher {
    pub fn new(allocator: Box<dyn StackBufferAllocator>) -> Self {
        Self {
            jmpbuf: MaybeUninit::uninit().as_mut_ptr(),
            allocator,
            stacks: Vec::new(),
            current_stack: 0,
        }
    }

    pub fn abort(&self) {
        unsafe {
            longjmp(self.jmpbuf, 1);
        }
    }

    #[inline(never)]
    pub fn run(&mut self) {
        let mut jmpbuf: MaybeUninit<jmp_buf> = MaybeUninit::uninit();
        unsafe {
            setjmp(jmpbuf.as_mut_ptr());
        }
        self.jmpbuf = jmpbuf.as_mut_ptr();

        // while we have a stack to use
        while let Some(stack) = self.next() {
            // set this stack as the current
            CURRENT.with(|current| {
                if let Ok(mut current) = current.try_borrow_mut() {
                    (*current) = Some(stack);
                } else {
                    panic!("failed to borrow current");
                }
            });

            // continue/launch the stack
            if let Some(_) = (*stack).bottom {
                let id = stack.id;

                stack.cont();

                self.stacks.retain(|x| x.id != id);
                if self.current_stack > 0 {
                    self.current_stack -= 1;
                }
            } else {
                stack.launch();
            }

            // set the current to none
            CURRENT.with(|current| {
                if let Ok(mut current) = current.try_borrow_mut() {
                    (*current) = None;
                } else {
                    panic!("failed to borrow current");
                }
            });
        }
    }

    pub fn next(&mut self) -> Option<&mut Stack> {
        if self.stacks.len() == 0 {
            return None;
        }
        let next_idx = (self.current_stack + 1) % self.stacks.len();
        let stack = self.stacks.get_mut(self.current_stack);
        self.current_stack = next_idx;
        stack
    }
}

pub trait StackBufferAllocator {
    fn alloc(&self, len: usize) -> StackBuffer;
    fn free(&self, stack_buffer: StackBuffer);
}

#[derive(Debug)]
pub struct StackBuffer {
    /// location
    loc: *mut c_void,
    /// length
    len: usize,
}

pub struct Stack {
    id: u64,
    dispatcher: *mut Dispatcher,
    pub bottom: Option<*mut c_void>,
    top: *mut c_void,
    stack_buffer: Option<StackBuffer>,
    continue_jmpbuf: *mut jmp_buf,
    program: Option<Box<dyn FnOnce(&mut Stack)>>,
}

impl Stack {
    pub fn new(id: u64, dispatcher: *mut Dispatcher, program: Box<dyn FnOnce(&mut Stack)>) -> Self {
        Self {
            id,
            dispatcher,
            bottom: None,
            top: std::ptr::null_mut(),
            stack_buffer: None,
            continue_jmpbuf: std::ptr::null_mut(),
            program: Some(program),
        }
    }

    #[inline(never)]
    pub fn stack_in(&mut self) {
        if let Some(stack_buffer) = &self.stack_buffer {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    stack_buffer.loc,
                    self.top,
                    self.bottom.unwrap().addr() - self.top.addr(),
                );
            }
        } else {
            panic!("stack_in: no stack buffer for stack {}", self.id);
        }
    }

    #[inline(never)]
    pub fn stack_out(&mut self) {
        let used = self.bottom.unwrap().addr() - self.top.addr();
        assert!(
            self.bottom.unwrap().addr() > self.top.addr(),
            "used must be greater than 0"
        );
        if let Some(stack_buffer) = self.stack_buffer.take() {
            if stack_buffer.len < used {
                unsafe {
                    (*self.dispatcher).allocator.free(stack_buffer);
                }
            } else {
                self.stack_buffer = Some(stack_buffer);
            }
        }
        if let None = &self.stack_buffer {
            // allocate a new stack buffer
            // eprintln!(
            //     "requesting stack buffer of size {} for stack {}",
            //     used, self.id
            // );
            self.stack_buffer = Some(unsafe { (*self.dispatcher).allocator.alloc(used) });
        }

        unsafe {
            std::ptr::copy_nonoverlapping(self.top, self.stack_buffer.as_ref().unwrap().loc, used);
        }
    }

    #[inline(never)]
    pub fn cont(&mut self) {
        self.stack_in();
        unsafe {
            longjmp(self.continue_jmpbuf, 1);
        }
    }

    #[inline(never)]
    pub fn launch(&mut self) {
        // pad the stack to prevent stack_in() from smashing the frame
        let mut padding = [0u8; 512];
        self.bottom = Some(padding.as_mut_ptr() as *mut c_void);
        if let Some(program) = self.program.take() {
            program(self);
        }
    }

    #[inline(never)]
    pub fn block(&mut self) {
        unsafe {
            let mut jmpbuf: MaybeUninit<jmp_buf> = MaybeUninit::uninit();
            if setjmp(jmpbuf.as_mut_ptr()) == 0 {
                self.continue_jmpbuf = jmpbuf.as_mut_ptr();
                self.top = self
                    .continue_jmpbuf
                    .with_addr(self.continue_jmpbuf.addr() - 32)
                    as *mut c_void;
                self.stack_out();
                longjmp((*self.dispatcher).jmpbuf, 1);
            }
        }
    }
}

struct HeapAllocator;

impl StackBufferAllocator for HeapAllocator {
    fn alloc(&self, len: usize) -> StackBuffer {
        let alloc = vec![7u8; len].into_boxed_slice();
        StackBuffer {
            loc: Box::leak(alloc).as_ptr() as *mut c_void,
            len,
        }
    }

    fn free(&self, stack_buffer: StackBuffer) {
        unsafe {
            drop(Box::from_raw(stack_buffer.loc));
        }
    }
}

fn test_coroutine(stack: &mut Stack, counter: *mut u64) {
    unsafe {
        loop {
            if *counter >= 200_000_000 {
                return;
            }
            *counter += 1;
            if *counter % 500_000 == 0 {
                eprintln!("{}: {}", stack.id, *counter);
            }
            stack.block();
        }
    }
}

fn main() {
    println!("Launching Dispatcher.");
    let mut dispatcher = Dispatcher::new(Box::new(HeapAllocator));
    let d = &mut dispatcher as *mut Dispatcher;
    let mut counter = Box::new(0u64);
    let counter_ptr = counter.as_mut() as *mut u64;
    for i in 0..10_000_000 {
        let stack = Stack::new(
            i,
            d,
            Box::new(move |stack| test_coroutine(stack, counter_ptr)),
        );
        dispatcher.stacks.push(stack);
    }
    dispatcher.run();
}
