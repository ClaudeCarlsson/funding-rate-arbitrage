pub mod shm_layout;

use anyhow::Result;
use shared_memory::ShmemConf;
use shm_layout::{RingBufferHeader, TickEvent};
use zerocopy::{AsBytes, FromBytes};
use std::sync::atomic::{AtomicU64, Ordering};

pub struct RustDisruptor {
    shmem: shared_memory::Shmem,
}

impl RustDisruptor {
    pub fn attach(name: &str) -> Result<Self> {
        let shmem = ShmemConf::new().os_id(name).open()?;
        Ok(Self { shmem })
    }

    /// Read ticks as a consumer.
    pub fn consume_ticks(&mut self) -> Vec<TickEvent> {
        let ptr = self.shmem.as_ptr();
        
        // Safety: We are mapping a C-compatible struct directly from memory.
        // We use volatile-style reads to ensure the compiler doesn't optimize away the shared memory polling.
        unsafe {
            let header = &mut *(ptr as *mut RingBufferHeader);
            
            let write_cursor = std::ptr::read_volatile(&header.write_cursor);
            let read_cursor = std::ptr::read_volatile(&header.read_cursor);
            let capacity = header.capacity;

            if read_cursor == write_cursor {
                return Vec::new();
            }

            let mut ticks = Vec::new();
            let mut current_read = read_cursor;

            while current_read < write_cursor {
                let index = (current_read % capacity) as usize;
                let offset = std::mem::size_of::<RingBufferHeader>() + (index * std::mem::size_of::<TickEvent>());
                
                let tick_ptr = ptr.add(offset) as *const TickEvent;
                let tick = std::ptr::read_volatile(tick_ptr);
                
                ticks.push(tick);
                current_read += 1;
            }

            // Update read cursor in shared memory
            std::ptr::write_volatile(&mut header.read_cursor, current_read);
            
            ticks
        }
    }
}
