use std::collections::HashMap;
use std::ffi::c_void;
use std::mem::MaybeUninit;

use setjmp::*;

type YieldValue = u128;

pub struct Dispatcher {
    jmpbuf: *mut jmp_buf,
    allocator: Box<dyn StackBufferAllocator>,
    stacks: Vec<Stack>,
    current_stack: usize,
    retrieved_values: HashMap<u64, YieldValue>,
}

impl Dispatcher {
    pub fn new(allocator: Box<dyn StackBufferAllocator>) -> Self {
        Self {
            #[allow(dangling_pointers_from_temporaries)]
            jmpbuf: MaybeUninit::uninit().as_mut_ptr(),
            allocator,
            stacks: Vec::new(),
            current_stack: 0,
            retrieved_values: HashMap::new(),
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

            // continue/launch the stack
            if let Some(_) = stack.bottom {
                stack.cont();
            } else {
                stack.launch();
            }
        }
    }

    fn next(&mut self) -> Option<&mut Stack> {
        if self.stacks.is_empty() {
            return None;
        }

        // Try each stack once, starting from current_stack
        let start = self.current_stack;
        loop {
            let current = self.current_stack;
            self.current_stack = (self.current_stack + 1) % self.stacks.len();

            let stack = &self.stacks[current];
            let (waiting_on, current_id) = (stack.waiting_on, stack.id);

            // If stack isn't waiting, it can run
            if waiting_on.is_none() {
                return Some(&mut self.stacks[current]);
            }

            let child_id = waiting_on.unwrap();

            // Check if child has yielded a value
            if let Some(child_idx) = self.stacks.iter().position(|s| s.id == child_id) {
                if let Some(value) = self.stacks[child_idx].value.take() {
                    self.retrieved_values.insert(current_id, value);
                    return Some(&mut self.stacks[current]);
                }
            }

            // Check if child is complete
            let child_exists = self.stacks.iter().any(|s| s.id == child_id && !s.is_complete);
            if !child_exists {
                let stack = &mut self.stacks[current];
                stack.waiting_on = None;
                return Some(stack);
            }

            // If we've tried all stacks and found nothing runnable, return None
            if self.current_stack == start {
                return None;
            }
        }
    }

    fn take_retrieved_value(&mut self, stack_id: u64) -> Option<YieldValue> {
        self.retrieved_values.remove(&stack_id)
    }

    fn next_id(&self) -> u64 {
        let max_id = self.stacks.iter()
            .map(|stack| stack.id)
            .max()
            .unwrap_or(0);
        max_id + 1
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
    value: Option<YieldValue>,
    waiting_on: Option<u64>,
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
            value: None,
            waiting_on: None,
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
        assert!(used > 0, "Stack underflow detected");
        assert!(used < 1024 * 1024, "Stack overflow detected"); // 1MB limit

        // Ensure we have enough buffer space
        if self.stack_buffer.as_ref().map_or(true, |buf| buf.len < used) {
            if let Some(old_buf) = self.stack_buffer.take() {
                unsafe {
                    (*self.dispatcher).allocator.free(old_buf);
                }
            }
            // Add extra padding to buffer
            self.stack_buffer = Some(unsafe {
                (*self.dispatcher).allocator.alloc(used + 1024)
            });
        }

        unsafe {
            let buffer = self.stack_buffer.as_ref().unwrap();
            std::ptr::copy_nonoverlapping(
                self.top,
                buffer.loc,
                used
            );
        }
    }

    #[inline(never)]
    pub fn cont(&mut self) {
        if self.is_complete {
            eprintln!(
                "Warning: Attempting to continue completed stack {}",
                self.id
            );
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
            // Make sure we return to dispatcher after completion
            unsafe {
                longjmp((*self.dispatcher).jmpbuf, 1);
            }
        }
    }

    #[inline(never)]
    pub fn block(&mut self) {
        unsafe {
            let mut jmpbuf: MaybeUninit<jmp_buf> = MaybeUninit::uninit();
            if setjmp(jmpbuf.as_mut_ptr()) == 0 {
                self.continue_jmpbuf = jmpbuf.as_mut_ptr();

                // Increase stack padding significantly
                const STACK_PADDING: usize = 1024;  // 1KB padding
                self.top = self
                    .continue_jmpbuf
                    .with_addr(self.continue_jmpbuf.addr() - STACK_PADDING)
                    as *mut c_void;

                // Ensure stack alignment
                self.top = self.top.with_addr(
                    (self.top.addr() + 15) & !15
                ) as *mut c_void;

                // Verify our stack invariants
                assert!(self.top.addr() % 16 == 0, "Stack must be 16-byte aligned");
                assert!(
                    self.bottom.unwrap().addr() > self.top.addr(),
                    "Stack overflow detected"
                );

                self.stack_out();
                longjmp((*self.dispatcher).jmpbuf, 1);
            }
        }
    }

    pub fn yield_value(&mut self, value: YieldValue) {
        self.value = Some(value);
        self.block();
    }

    pub fn get_yielded(&mut self, child_id: u64) -> Option<YieldValue> {
        unsafe {
            // First check if we already have a value
            if let Some(value) = (*self.dispatcher).take_retrieved_value(self.id) {
                return Some(value);
            }

            // No value yet, set waiting and block
            self.waiting_on = Some(child_id);

            // Keep blocking until we either get a value or child completes
            loop {
                self.block();

                // After each block, check for value first
                if let Some(value) = (*self.dispatcher).take_retrieved_value(self.id) {
                    self.waiting_on = None;
                    return Some(value);
                }

                // Then check if child still exists
                let child_exists = (*self.dispatcher).stacks.iter()
                    .any(|s| s.id == child_id && !s.is_complete);

                if !child_exists {
                    self.waiting_on = None;
                    return None;
                }

                // Still waiting for child
            }
        }
    }

    pub fn spawn<F>(&mut self, f: F) -> u64
    where
        F: FnOnce(&mut Stack) + 'static
    {
        unsafe {
            let new_id = (*self.dispatcher).next_id();
            let new_stack = Stack::new(new_id, self.dispatcher, Box::new(f));
            (*self.dispatcher).stacks.push(new_stack);
            new_id
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
    allocations: std::sync::atomic::AtomicUsize,
}

impl HeapAllocator {
    fn new() -> Self {
        Self {
            allocations: std::sync::atomic::AtomicUsize::new(0),
        }
    }
}

impl StackBufferAllocator for HeapAllocator {
    fn alloc(&self, len: usize) -> StackBuffer {
        let len = (len + 15) & !15;
        let layout = std::alloc::Layout::from_size_align(len, 16).expect("Failed to create layout");
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }

        self.allocations.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        StackBuffer {
            loc: ptr as *mut c_void,
            len,
        }
    }

    fn free(&self, stack_buffer: StackBuffer) {
        let current = self.allocations.load(std::sync::atomic::Ordering::SeqCst);
        if current == 0 {
            eprintln!("Warning: Attempting to free buffer when allocation count is 0");
            return;
        }

        self.allocations.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);

        let len = (stack_buffer.len + 15) & !15;
        unsafe {
            let layout = std::alloc::Layout::from_size_align(len, 16).expect("Failed to create layout");
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
                    eprintln!(
                        "stack {} | local:{} counter:{} vec_size:{} (DONE)",
                        stack.id,
                        local,
                        *counter,
                        growing_vec.len()
                    );
                }
                return;
            } else if *counter % 10_000 == 0 {
                eprintln!(
                    "stack {} | local:{} counter:{} vec_size:{}",
                    stack.id,
                    local,
                    *counter,
                    growing_vec.len()
                );
            }

            growing_vec.push(runs);
            local += *counter / 2;
            *counter += 1;
            runs += 1;
            stack.block();
        }
    }
}

fn number_sequence(stack: &mut Stack) {
    println!("Starting number_sequence");
    for i in 0..5 {
        println!("Yielding value: {}", i);
        stack.yield_value(i as YieldValue);
        println!("After yield: {}", i);
    }
    println!("Ending number_sequence");
}

fn parent_routine(stack: &mut Stack) {
    println!("Starting parent_routine");
    let child_id = stack.spawn(number_sequence);
    println!("Spawned child with id: {}", child_id);

    while let Some(value) = stack.get_yielded(child_id) {
        println!("Parent received value: {}", value);
    }
    println!("Parent ending");
}

fn main() {
    println!("Launching Dispatcher.");
    let mut dispatcher = Dispatcher::new(Box::new(HeapAllocator::new()));
    let d = &mut dispatcher as *mut Dispatcher;

    let stack = Stack::new(0, d, Box::new(parent_routine));
    dispatcher.stacks.push(stack);
    dispatcher.run();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_yield_sequence() {
        let mut dispatcher = Dispatcher::new(Box::new(HeapAllocator::new()));
        let d = &mut dispatcher as *mut Dispatcher;

        let stack = Stack::new(0, d, Box::new(|stack| {
            for i in 0..5 {
                stack.yield_value(i as YieldValue);
            }
        }));
        dispatcher.stacks.push(stack);
        dispatcher.run();
    }

    #[test]
    fn test_parent_child_communication() {
        let mut dispatcher = Dispatcher::new(Box::new(HeapAllocator::new()));
        let d = &mut dispatcher as *mut Dispatcher;

        let stack = Stack::new(0, d, Box::new(|stack| {
            let child_id = stack.spawn(|child| {
                for i in 0..5 {
                    child.yield_value((i * 2) as YieldValue);
                }
            });

            let mut sum = 0;
            while let Some(value) = stack.get_yielded(child_id) {
                sum += value;
            }
            assert_eq!(sum, 20); // 0 + 2 + 4 + 6 + 8
        }));
        dispatcher.stacks.push(stack);
        dispatcher.run();
    }

    #[test]
    fn test_multiple_children() {
        let mut dispatcher = Dispatcher::new(Box::new(HeapAllocator::new()));
        let d = &mut dispatcher as *mut Dispatcher;

        let stack = Stack::new(0, d, Box::new(|stack| {
            // Spawn two children
            let child1_id = stack.spawn(|child| {
                for i in 0..3 {
                    child.yield_value(i as YieldValue);
                }
            });

            let child2_id = stack.spawn(|child| {
                for i in 0..3 {
                    child.yield_value((i + 10) as YieldValue);
                }
            });

            // Collect values from both children alternately
            let mut values1 = Vec::new();
            let mut values2 = Vec::new();
            let mut done1 = false;
            let mut done2 = false;

            while !done1 || !done2 {
                if !done1 {
                    if let Some(value) = stack.get_yielded(child1_id) {
                        values1.push(value);
                    } else {
                        done1 = true;
                    }
                }
                if !done2 {
                    if let Some(value) = stack.get_yielded(child2_id) {
                        values2.push(value);
                    } else {
                        done2 = true;
                    }
                }
            }

            assert_eq!(values1, vec![0, 1, 2]);
            assert_eq!(values2, vec![10, 11, 12]);
        }));
        dispatcher.stacks.push(stack);
        dispatcher.run();
    }

    #[test]
    fn test_nested_coroutines() {
        let mut dispatcher = Dispatcher::new(Box::new(HeapAllocator::new()));
        let d = &mut dispatcher as *mut Dispatcher;

        let stack = Stack::new(0, d, Box::new(|stack| {
            let child_id = stack.spawn(|child| {
                // Spawn a grandchild from the child
                let grandchild_id = child.spawn(|grandchild| {
                    for i in 0..3 {
                        grandchild.yield_value(i as YieldValue);
                    }
                });

                // Child modifies and forwards values from grandchild
                while let Some(value) = child.get_yielded(grandchild_id) {
                    child.yield_value(value * 2);
                }
            });

            let mut values = Vec::new();
            while let Some(value) = stack.get_yielded(child_id) {
                values.push(value);
            }
            assert_eq!(values, vec![0, 2, 4]);
        }));
        dispatcher.stacks.push(stack);
        dispatcher.run();
    }

    #[test]
    fn test_stack_growth() {
        let mut dispatcher = Dispatcher::new(Box::new(HeapAllocator::new()));
        let d = &mut dispatcher as *mut Dispatcher;

        let stack = Stack::new(0, d, Box::new(|stack| {
            let child_id = stack.spawn(|s| {
                // Function to measure current stack size and print details
                fn get_stack_info(s: &Stack) -> (usize, String) {
                    if let Some(bottom) = s.bottom {
                        let stack_marker = 0u8; // Local variable to get current stack position
                        let current_pos = &stack_marker as *const u8 as *mut c_void;
                        let size = bottom.addr() - current_pos.addr();
                        let info = format!(
                            "bottom: {:p}, current: {:p}, size: {} bytes",
                            bottom,
                            current_pos,
                            size
                        );
                        (size, info)
                    } else {
                        (0, "no bottom pointer".to_string())
                    }
                }

                // Create progressively larger stack allocations
                for i in 0..4 {
                    // Print stack info before allocation
                    let (initial_size, initial_info) = get_stack_info(s);
                    println!("Before allocation {}: {}", i, initial_info);

                    // Recursive function to create stack frames and measure stack
                    fn recursive_stack_growth(s: &Stack, depth: usize, target: usize, array: &mut [u8; 256]) -> usize {
                        // Measure current stack size at this depth
                        let (current_size, info) = get_stack_info(s);
                        println!("  At depth {}: {}", depth, info);

                        if depth < target {
                            // Modify array to prevent optimization
                            for i in 0..array.len() {
                                array[i] = array[i].wrapping_add(1);
                            }
                            // Recurse and return deepest stack size
                            recursive_stack_growth(s, depth + 1, target, array)
                        } else {
                            current_size // Return stack size at maximum depth
                        }
                    }

                    // Create stack growth through recursion
                    let mut array = [0u8; 256];
                    let depth = 1 << i;  // 1, 2, 4, 8 stack frames
                    let max_stack_size = recursive_stack_growth(s, 0, depth, &mut array);

                    // Print stack info after allocation
                    let (final_size, final_info) = get_stack_info(s);
                    println!("After allocation {}: {}", i, final_info);
                    println!("Created {} stack frames, max stack size: {} bytes (grew by: {} bytes)",
                        depth,
                        max_stack_size,
                        max_stack_size.saturating_sub(initial_size));

                    // Yield maximum stack size seen
                    s.yield_value(max_stack_size as YieldValue);
                }
            });

            // Collect and verify the yielded values
            let mut sizes = Vec::new();
            while let Some(size) = stack.get_yielded(child_id) {
                println!("\nStack measurement: {} bytes", size);
                sizes.push(size);
            }

            // Print all collected sizes
            println!("\nAll stack sizes: {:?}", sizes);

            // Verify we got 4 measurements
            assert_eq!(sizes.len(), 4, "Should have 4 stack measurements");
        }));

        dispatcher.stacks.push(stack);
        dispatcher.run();
    }
}

