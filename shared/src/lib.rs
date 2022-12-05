#![allow(clippy::all)]

use compute_engine::{BaseEngine, ComputeEngine};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::{AutoCommandBufferBuilder, CopyBufferInfo},
};

#[cfg(test)]
mod tests;

pub fn entrypoint() {
    // Start Compute Engine
    let compute_engine = ComputeEngine::new();

    // Print some information
    ComputeEngine::print_api_information(compute_engine.get_instance(), log::Level::Info);

    // Source Buffer
    let source_content: Vec<i32> = (0..64).collect();
    let source_buffer = CpuAccessibleBuffer::from_iter(
        compute_engine.get_logical_device().get_device(),
        BufferUsage {
            transfer_src: true,
            ..Default::default()
        },
        false,
        source_content,
    )
    .expect("failed to create source buffer");

    // Destination Buffer
    let destination_content: Vec<i32> = (0..64).map(|_| 0).collect();
    let destination_buffer = CpuAccessibleBuffer::from_iter(
        compute_engine.get_logical_device().get_device(),
        BufferUsage {
            transfer_dst: true,
            ..Default::default()
        },
        false,
        destination_content,
    )
    .expect("failed to create destination buffer");

    // Submit Command Buffer for Computation
    compute_engine.compute(&|engine: &ComputeEngine| {
        let mut builder = AutoCommandBufferBuilder::primary(
            engine.get_logical_device().get_device(),
            engine.get_logical_device().get_queue_family_index(),
            vulkano::command_buffer::CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .copy_buffer(CopyBufferInfo::buffers(
                source_buffer.clone(),
                destination_buffer.clone(),
            ))
            .unwrap();

        builder.build().unwrap()
    });

    // Assert results
    let source_content = source_buffer.read().unwrap();
    let destination_content = destination_buffer.read().unwrap();

    assert_eq!(&*source_content, &*destination_content);
    log::info!("Assertion passed");
}
