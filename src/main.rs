#![feature(strict_provenance)]

use setjmp::*;
use std::cell::RefCell;
use std::ffi::c_void;
use std::mem::MaybeUninit;
use std::thread_local;

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

    pub fn block(&self) {
        CURRENT.with(|current| unsafe {
            if let Ok(mut current) = current.try_borrow_mut() {
                if let Some(current) = current.as_mut() {
                    let current = *current;
                    let mut jmpbuf: MaybeUninit<jmp_buf> = MaybeUninit::uninit();
                    if setjmp(jmpbuf.as_mut_ptr()) == 0 {
                        (*current).continue_jmpbuf = jmpbuf.as_mut_ptr();
                        (*current).top = (*current)
                            .continue_jmpbuf
                            .with_addr((*current).continue_jmpbuf.addr() - 32)
                            as *mut c_void;
                        longjmp(self.jmpbuf, 1);
                    }
                }
            }
        });
    }

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
                }
            });

            // continue/launch the stack
            if let Some(_) = (*stack).bottom {
                stack.cont();
            } else {
                stack.launch();
            }

            // set the current to none
            CURRENT.with(|current| {
                if let Ok(mut current) = current.try_borrow_mut() {
                    (*current) = None;
                }
            });
        }
    }

    pub fn next(&mut self) -> Option<&mut Stack> {
        let next_idx = (self.current_stack + 1) % self.stacks.len();
        let stack = self.stacks.get_mut(self.current_stack);
        self.current_stack = next_idx;
        stack
    }

    pub fn launch(&mut self, _stack: &mut Stack) {
        println!("start stack");
        for i in 0..10 {
            println!("{}", i);
            self.block();
        }
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
    pub bottom: Option<*mut c_void>,
    top: *mut c_void,
    stack_buffer: Option<StackBuffer>,
    continue_jmpbuf: *mut jmp_buf,
}

impl Stack {
    pub fn new() -> Self {
        Self {
            bottom: None,
            top: std::ptr::null_mut(),
            stack_buffer: None,
            continue_jmpbuf: std::ptr::null_mut(),
        }
    }

    pub fn stack_in(&mut self) {
        if let Some(stack_buffer) = &self.stack_buffer {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.top,
                    stack_buffer.loc,
                    self.bottom.unwrap_unchecked().addr() - self.top.addr(),
                );
            }
        }
    }

    pub fn stack_out(&mut self, dispatcher: &Dispatcher) {
        let used = unsafe { self.bottom.unwrap_unchecked() }.addr() - self.top.addr();
        if let Some(stack_buffer) = self.stack_buffer.take() {
            if stack_buffer.len < used {
                dispatcher.allocator.free(stack_buffer);
            } else {
                self.stack_buffer = Some(stack_buffer);
            }
        }
        if let None = &self.stack_buffer {
            self.stack_buffer = Some(dispatcher.allocator.alloc(used));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.stack_buffer.as_ref().unwrap_unchecked().loc,
                self.top,
                used,
            );
        }
    }

    pub fn cont(&mut self) {
        self.stack_in();
        unsafe {
            longjmp(self.continue_jmpbuf, 1);
        }
    }

    pub fn launch(&mut self) {
        // pad the stack to prevent stack_in() from smashing the frame
        let mut padding = [0u8; 300];
        self.bottom = Some(padding.as_mut_ptr() as *mut c_void);
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

fn main() {
    println!("Launching Dispatcher.");
    let mut dispatcher = Dispatcher::new(Box::new(HeapAllocator));
    dispatcher.run();
}
