#![feature(strict_provenance)]

use setjmp::*;
use std::cell::RefCell;
use std::ffi::c_void;
use std::mem::MaybeUninit;
use std::thread_local;

thread_local! {
    static CURRENT: RefCell<Option<Stack>> = RefCell::new(None);
}

struct Dispatcher {
    jmpbuf: *mut jmp_buf,
}

struct Stack {
    dispatcher: *mut Dispatcher,
    bottom: *mut c_void,
    top: *mut c_void,
    stack_buffer: *mut c_void,
    continue_jmpbuf: *mut jmp_buf,
}

fn runtime_block() {
    with_current_stack(|current| unsafe {
        let mut jmpbuf: MaybeUninit<jmp_buf> = MaybeUninit::uninit();
        if setjmp(jmpbuf.as_mut_ptr()) == 0 {
            current.continue_jmpbuf = jmpbuf.as_mut_ptr();
            current.top = current
                .continue_jmpbuf
                .with_addr(current.continue_jmpbuf.addr() - 32)
                as *mut c_void;
            longjmp_s((*current.dispatcher).jmpbuf);
        }
    });
}

fn runtime_abort() {
    with_current_stack(|current| unsafe {
        longjmp_s((*current.dispatcher).jmpbuf);
    });
}

unsafe fn longjmp_s(jmpbuf: *mut jmp_buf) {
    longjmp(jmpbuf, 1);
    panic!("longjmp returned!");
}

fn with_current_stack(f: impl FnOnce(&mut Stack)) {
    CURRENT.with(|current| {
        if let Ok(mut current) = current.try_borrow_mut() {
            if let Some(current) = current.as_mut() {
                f(current);
            }
        }
    });
}

fn main() {
    println!("Hello, world!");
}
