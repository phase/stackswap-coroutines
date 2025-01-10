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
            #[allow(dangling_pointers_from_temporaries)]
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
            if stack.is_complete {
                self.stacks.remove(self.current_stack);
                if self.current_stack >= self.stacks.len() && !self.stacks.is_empty() {
                    self.current_stack = 0;
                }
                continue;
            }
            // set this stack as the current
            // CURRENT.with(|current| {
            //     if let Ok(mut current) = current.try_borrow_mut() {
            //         (*current) = Some(stack);
            //     } else {
            //         panic!("failed to borrow current");
            //     }
            // });

            // continue/launch the stack
            if let Some(_) = (*stack).bottom {
                stack.cont();
            } else {
                stack.launch();
            }

            // set the current to none
            // CURRENT.with(|current| {
            //     if let Ok(mut current) = current.try_borrow_mut() {
            //         (*current) = None;
            //     } else {
            //         panic!("failed to borrow current");
            //     }
            // });
        }
    }

    pub fn next(&mut self) -> Option<&mut Stack> {
        if self.stacks.is_empty() {
            return None;
        }
        if self.stacks.len() > 1 {
            self.current_stack = (self.current_stack + 1) % self.stacks.len();
        }
        self.stacks.get_mut(self.current_stack)
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
    is_complete: bool,
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
            is_complete: false,
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
                self.stack_buffer = None;
            } else {
                self.stack_buffer = Some(stack_buffer);
            }
        }

        if self.stack_buffer.is_none() {
            self.stack_buffer = Some(unsafe { (*self.dispatcher).allocator.alloc(used) });
        }

        unsafe {
            let buffer = self.stack_buffer.as_ref().unwrap();
            std::ptr::copy_nonoverlapping(self.top, buffer.loc, used);
        }
    }

    #[inline(never)]
    pub fn cont(&mut self) {
        if self.is_complete {
            eprintln!("Warning: Attempting to continue completed stack {}", self.id);
            return;
        }
        self.stack_in();
        unsafe {
            longjmp(self.continue_jmpbuf, 1);
        }
    }

    #[inline(never)]
    pub fn launch(&mut self) {
        let mut padding = [0u8; 64];
        self.bottom = Some(padding.as_mut_ptr() as *mut c_void);
        if let Some(program) = self.program.take() {
            program(self);
            self.bottom = None;
            self.is_complete = true;
        }
    }

    #[inline(never)]
    pub fn block(&mut self) {
        unsafe {
            let mut jmpbuf: MaybeUninit<jmp_buf> = MaybeUninit::uninit();
            if setjmp(jmpbuf.as_mut_ptr()) == 0 {
                self.continue_jmpbuf = jmpbuf.as_mut_ptr();
                // Increase the offset to avoid potential stack corruption
                self.top = self
                    .continue_jmpbuf
                    .with_addr(self.continue_jmpbuf.addr() - 64)
                    as *mut c_void;
                // Save stack state
                self.stack_out();

                // Jump back to dispatcher
                longjmp((*self.dispatcher).jmpbuf, 1);
            }
            // We've returned from a continuation, no need to restore stack
            // as it was already restored in cont()
        }
    }
}

impl Drop for Stack {
    fn drop(&mut self) {
        if let Some(buffer) = self.stack_buffer.take() {
            unsafe {
                (*self.dispatcher).allocator.free(buffer);
            }
        }
    }
}

struct HeapAllocator {
    #[allow(dead_code)]
    allocations: std::sync::atomic::AtomicUsize,
}

impl HeapAllocator {
    fn new() -> Self {
        Self {
            allocations: std::sync::atomic::AtomicUsize::new(0)
        }
    }
}

impl StackBufferAllocator for HeapAllocator {
    fn alloc(&self, len: usize) -> StackBuffer {
        let len = (len + 15) & !15;
        let layout = std::alloc::Layout::from_size_align(len, 16)
            .expect("Failed to create layout");
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }

        let _count = self.allocations.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        // eprintln!("Allocating buffer {} at {:p}", count + 1, ptr);

        StackBuffer {
            loc: ptr as *mut c_void,
            len,
        }
    }

    fn free(&self, stack_buffer: StackBuffer) {
        let current = self.allocations.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);

        // eprintln!("Freeing buffer {} at {:p}", current, stack_buffer.loc);
        if current == 0 {
            eprintln!("Warning: Attempting to free buffer at {:p} when allocation count is 0",
                     stack_buffer.loc);
            self.allocations.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            return;
        }

        let len = (stack_buffer.len + 15) & !15;
        unsafe {
            let layout = std::alloc::Layout::from_size_align(len, 16)
                .expect("Failed to create layout");
            std::alloc::dealloc(stack_buffer.loc as *mut u8, layout);
        }
    }
}

fn test_coroutine(stack: &mut Stack, counter: *mut u64) {
    let mut runs = 0u64;
    let mut local = 0u64;
    let mut growing_vec = Vec::new();

    unsafe {
        loop {
            if *counter >= 1_000_000 {
                *counter += 1;
                if *counter % 1_000 == 0 {
                    eprintln!("stack {} | local:{} counter:{} vec_size:{} (DONE)",
                        stack.id, local, *counter, growing_vec.len());
                }
                return;
            } else if *counter % 10_000 == 0 {
                eprintln!("stack {} | local:{} counter:{} vec_size:{}",
                    stack.id, local, *counter, growing_vec.len());
            }

            growing_vec.push(runs);
            local += *counter / 2;
            *counter += 1;
            runs += 1;
            stack.block();
        }
    }
}

fn main() {
    println!("Launching Dispatcher.");
    let mut dispatcher = Dispatcher::new(Box::new(HeapAllocator::new()));
    let d = &mut dispatcher as *mut Dispatcher;
    let mut counter = Box::new(0u64);
    let counter_ptr = counter.as_mut() as *mut u64;
    for i in 0..24_351 {
        let stack = Stack::new(
            i,
            d,
            Box::new(move |stack| test_coroutine(stack, counter_ptr)),
        );
        dispatcher.stacks.push(stack);
    }
    dispatcher.run();
}
