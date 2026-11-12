use core::ffi::{c_char, c_void};

use super::*;

#[cfg_attr(
    target_os = "linux",
    link(name = "libvulkan.so.1", kind = "dylib", modifiers = "+verbatim")
)]
#[cfg_attr(target_os = "android", link(name = "vulkan"))]
#[cfg_attr(target_os = "windows", link(name = "vulkan-1", kind = "raw-dylib"))]
unsafe extern "system" {
    #[cfg(target_os = "linux")]
    pub(in crate::vulkan) fn vkCreateXlibSurfaceKHR(
        instance: *mut c_void,
        pCreateInfo: *const VkXlibSurfaceCreateInfoKHR,
        pAllocator: *const c_void,
        pSurface: *mut u64,
    ) -> i32;
    #[cfg(target_os = "windows")]
    pub(in crate::vulkan) fn vkCreateWin32SurfaceKHR(
        instance: *mut c_void,
        pCreateInfo: *const VkWin32SurfaceCreateInfoKHR,
        pAllocator: *const c_void,
        pSurface: *mut u64,
    ) -> i32;
    pub(in crate::vulkan) fn vkDestroySurfaceKHR(
        instance: *mut c_void,
        surface: u64,
        pAllocator: *const c_void,
    );
    pub(in crate::vulkan) fn vkGetPhysicalDeviceSurfaceSupportKHR(
        physicalDevice: *mut c_void,
        queueFamilyIndex: u32,
        surface: u64,
        pSupported: *mut u32,
    ) -> i32;
    pub(in crate::vulkan) fn vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
        physicalDevice: *mut c_void,
        surface: u64,
        pSurfaceCapabilities: *mut VkSurfaceCapabilitiesKHR,
    ) -> i32;
    pub(in crate::vulkan) fn vkGetPhysicalDeviceSurfaceFormatsKHR(
        physicalDevice: *mut c_void,
        surface: u64,
        pSurfaceFormatCount: *mut u32,
        pSurfaceFormats: *mut VkSurfaceFormatKHR,
    ) -> i32;
    pub(in crate::vulkan) fn vkCreateSwapchainKHR(
        device: *mut c_void,
        pCreateInfo: *const VkSwapchainCreateInfoKHR,
        pAllocator: *const c_void,
        pSwapchain: *mut u64,
    ) -> i32;
    pub(in crate::vulkan) fn vkDestroySwapchainKHR(
        device: *mut c_void,
        swapchain: u64,
        pAllocator: *const c_void,
    );
    pub(in crate::vulkan) fn vkGetSwapchainImagesKHR(
        device: *mut c_void,
        swapchain: u64,
        pSwapchainImageCount: *mut u32,
        pSwapchainImages: *mut u64,
    ) -> i32;
    pub(in crate::vulkan) fn vkAcquireNextImageKHR(
        device: *mut c_void,
        swapchain: u64,
        timeout: u64,
        semaphore: u64,
        fence: u64,
        pImageIndex: *mut u32,
    ) -> i32;
    pub(in crate::vulkan) fn vkQueuePresentKHR(
        queue: *mut c_void,
        pPresentInfo: *const VkPresentInfoKHR,
    ) -> i32;
    pub(in crate::vulkan) fn vkCreateFence(
        device: *mut c_void,
        pCreateInfo: *const VkFenceCreateInfo,
        pAllocator: *const c_void,
        pFence: *mut u64,
    ) -> i32;
    pub(in crate::vulkan) fn vkDestroyFence(
        device: *mut c_void,
        fence: u64,
        pAllocator: *const c_void,
    );
    pub(in crate::vulkan) fn vkWaitForFences(
        device: *mut c_void,
        fenceCount: u32,
        pFences: *const u64,
        waitAll: u32,
        timeout: u64,
    ) -> i32;
    pub(in crate::vulkan) fn vkResetFences(
        device: *mut c_void,
        fenceCount: u32,
        pFences: *const u64,
    ) -> i32;
    pub(in crate::vulkan) fn vkCreateSemaphore(
        device: *mut c_void,
        pCreateInfo: *const VkSemaphoreCreateInfo,
        pAllocator: *const c_void,
        pSemaphore: *mut u64,
    ) -> i32;
    pub(in crate::vulkan) fn vkDestroySemaphore(
        device: *mut c_void,
        semaphore: u64,
        pAllocator: *const c_void,
    );
    pub(in crate::vulkan) fn vkResetCommandBuffer(commandBuffer: *mut c_void, flags: u32) -> i32;
    pub(in crate::vulkan) fn vkCmdPushConstants(
        commandBuffer: *mut c_void,
        layout: u64,
        stageFlags: u32,
        offset: u32,
        size: u32,
        pValues: *const c_void,
    );
    pub(in crate::vulkan) fn vkCmdCopyBuffer(
        commandBuffer: *mut c_void,
        srcBuffer: u64,
        dstBuffer: u64,
        regionCount: u32,
        pRegions: *const VkBufferCopy,
    );
    pub(in crate::vulkan) fn vkCreateQueryPool(
        device: *mut c_void,
        pCreateInfo: *const VkQueryPoolCreateInfo,
        pAllocator: *const c_void,
        pQueryPool: *mut u64,
    ) -> i32;
    pub(in crate::vulkan) fn vkDestroyQueryPool(
        device: *mut c_void,
        queryPool: u64,
        pAllocator: *const c_void,
    );
    pub(in crate::vulkan) fn vkCmdWriteTimestamp(
        commandBuffer: *mut c_void,
        pipelineStage: i32,
        queryPool: u64,
        query: u32,
    );
    pub(in crate::vulkan) fn vkCmdResetQueryPool(
        commandBuffer: *mut c_void,
        queryPool: u64,
        firstQuery: u32,
        queryCount: u32,
    );
    pub(in crate::vulkan) fn vkGetQueryPoolResults(
        device: *mut c_void,
        queryPool: u64,
        firstQuery: u32,
        queryCount: u32,
        dataSize: usize,
        pData: *mut c_void,
        stride: u64,
        flags: u32,
    ) -> i32;
    pub(in crate::vulkan) fn vkCreateInstance(
        pCreateInfo: *const VkInstanceCreateInfo,
        pAllocator: *const c_void,
        pInstance: *mut *mut c_void,
    ) -> i32;
    pub(in crate::vulkan) fn vkDestroyInstance(instance: *mut c_void, pAllocator: *const c_void);
    pub(in crate::vulkan) fn vkEnumeratePhysicalDevices(
        instance: *mut c_void,
        pPhysicalDeviceCount: *mut u32,
        pPhysicalDevices: *mut *mut c_void,
    ) -> i32;
    pub(in crate::vulkan) fn vkGetPhysicalDeviceQueueFamilyProperties(
        physicalDevice: *mut c_void,
        pQueueFamilyPropertyCount: *mut u32,
        pQueueFamilyProperties: *mut VkQueueFamilyProperties,
    );
    pub(in crate::vulkan) fn vkGetPhysicalDeviceMemoryProperties(
        physicalDevice: *mut c_void,
        pMemoryProperties: *mut VkPhysicalDeviceMemoryProperties,
    );
    pub(in crate::vulkan) fn vkGetPhysicalDeviceProperties(
        physicalDevice: *mut c_void,
        pProperties: *mut VkPhysicalDeviceProperties,
    );
    pub(in crate::vulkan) fn vkGetPhysicalDeviceFormatProperties(
        physicalDevice: *mut c_void,
        format: i32,
        pFormatProperties: *mut VkFormatProperties,
    );
    pub(in crate::vulkan) fn vkCreateDevice(
        physicalDevice: *mut c_void,
        pCreateInfo: *const VkDeviceCreateInfo,
        pAllocator: *const c_void,
        pDevice: *mut *mut c_void,
    ) -> i32;
    pub(in crate::vulkan) fn vkDestroyDevice(device: *mut c_void, pAllocator: *const c_void);
    pub(in crate::vulkan) fn vkGetDeviceQueue(
        device: *mut c_void,
        queueFamilyIndex: u32,
        queueIndex: u32,
        pQueue: *mut *mut c_void,
    );
    pub(in crate::vulkan) fn vkDeviceWaitIdle(device: *mut c_void) -> i32;
    pub(in crate::vulkan) fn vkQueueSubmit(
        queue: *mut c_void,
        submitCount: u32,
        pSubmits: *const VkSubmitInfo,
        fence: u64,
    ) -> i32;
    pub(in crate::vulkan) fn vkQueueWaitIdle(queue: *mut c_void) -> i32;
    pub(in crate::vulkan) fn vkCreateBuffer(
        device: *mut c_void,
        pCreateInfo: *const VkBufferCreateInfo,
        pAllocator: *const c_void,
        pBuffer: *mut u64,
    ) -> i32;
    pub(in crate::vulkan) fn vkDestroyBuffer(
        device: *mut c_void,
        buffer: u64,
        pAllocator: *const c_void,
    );
    pub(in crate::vulkan) fn vkGetBufferMemoryRequirements(
        device: *mut c_void,
        buffer: u64,
        pMemoryRequirements: *mut VkMemoryRequirements,
    );
    pub(in crate::vulkan) fn vkCreateImage(
        device: *mut c_void,
        pCreateInfo: *const VkImageCreateInfo,
        pAllocator: *const c_void,
        pImage: *mut u64,
    ) -> i32;
    pub(in crate::vulkan) fn vkDestroyImage(
        device: *mut c_void,
        image: u64,
        pAllocator: *const c_void,
    );
    pub(in crate::vulkan) fn vkGetImageMemoryRequirements(
        device: *mut c_void,
        image: u64,
        pMemoryRequirements: *mut VkMemoryRequirements,
    );
    pub(in crate::vulkan) fn vkAllocateMemory(
        device: *mut c_void,
        pAllocateInfo: *const VkMemoryAllocateInfo,
        pAllocator: *const c_void,
        pMemory: *mut u64,
    ) -> i32;
    pub(in crate::vulkan) fn vkFreeMemory(
        device: *mut c_void,
        memory: u64,
        pAllocator: *const c_void,
    );
    pub(in crate::vulkan) fn vkBindBufferMemory(
        device: *mut c_void,
        buffer: u64,
        memory: u64,
        memoryOffset: u64,
    ) -> i32;
    pub(in crate::vulkan) fn vkBindImageMemory(
        device: *mut c_void,
        image: u64,
        memory: u64,
        memoryOffset: u64,
    ) -> i32;
    pub(in crate::vulkan) fn vkMapMemory(
        device: *mut c_void,
        memory: u64,
        offset: u64,
        size: u64,
        flags: u32,
        ppData: *mut *mut c_void,
    ) -> i32;
    pub(in crate::vulkan) fn vkUnmapMemory(device: *mut c_void, memory: u64);
    pub(in crate::vulkan) fn vkCreateImageView(
        device: *mut c_void,
        pCreateInfo: *const VkImageViewCreateInfo,
        pAllocator: *const c_void,
        pView: *mut u64,
    ) -> i32;
    pub(in crate::vulkan) fn vkDestroyImageView(
        device: *mut c_void,
        imageView: u64,
        pAllocator: *const c_void,
    );
    pub(in crate::vulkan) fn vkCreateRenderPass(
        device: *mut c_void,
        pCreateInfo: *const VkRenderPassCreateInfo,
        pAllocator: *const c_void,
        pRenderPass: *mut u64,
    ) -> i32;
    pub(in crate::vulkan) fn vkDestroyRenderPass(
        device: *mut c_void,
        renderPass: u64,
        pAllocator: *const c_void,
    );
    pub(in crate::vulkan) fn vkCreateFramebuffer(
        device: *mut c_void,
        pCreateInfo: *const VkFramebufferCreateInfo,
        pAllocator: *const c_void,
        pFramebuffer: *mut u64,
    ) -> i32;
    pub(in crate::vulkan) fn vkDestroyFramebuffer(
        device: *mut c_void,
        framebuffer: u64,
        pAllocator: *const c_void,
    );
    pub(in crate::vulkan) fn vkCreateShaderModule(
        device: *mut c_void,
        pCreateInfo: *const VkShaderModuleCreateInfo,
        pAllocator: *const c_void,
        pShaderModule: *mut u64,
    ) -> i32;
    pub(in crate::vulkan) fn vkDestroyShaderModule(
        device: *mut c_void,
        shaderModule: u64,
        pAllocator: *const c_void,
    );
    pub(in crate::vulkan) fn vkCreatePipelineCache(
        device: *mut c_void,
        pCreateInfo: *const VkPipelineCacheCreateInfo,
        pAllocator: *const c_void,
        pPipelineCache: *mut u64,
    ) -> i32;
    pub(in crate::vulkan) fn vkDestroyPipelineCache(
        device: *mut c_void,
        pipelineCache: u64,
        pAllocator: *const c_void,
    );
    pub(in crate::vulkan) fn vkCreatePipelineLayout(
        device: *mut c_void,
        pCreateInfo: *const VkPipelineLayoutCreateInfo,
        pAllocator: *const c_void,
        pPipelineLayout: *mut u64,
    ) -> i32;
    pub(in crate::vulkan) fn vkDestroyPipelineLayout(
        device: *mut c_void,
        pipelineLayout: u64,
        pAllocator: *const c_void,
    );
    pub(in crate::vulkan) fn vkCreateGraphicsPipelines(
        device: *mut c_void,
        pipelineCache: u64,
        createInfoCount: u32,
        pCreateInfos: *const VkGraphicsPipelineCreateInfo,
        pAllocator: *const c_void,
        pPipelines: *mut u64,
    ) -> i32;
    pub(in crate::vulkan) fn vkDestroyPipeline(
        device: *mut c_void,
        pipeline: u64,
        pAllocator: *const c_void,
    );
    pub(in crate::vulkan) fn vkCreateCommandPool(
        device: *mut c_void,
        pCreateInfo: *const VkCommandPoolCreateInfo,
        pAllocator: *const c_void,
        pCommandPool: *mut u64,
    ) -> i32;
    pub(in crate::vulkan) fn vkDestroyCommandPool(
        device: *mut c_void,
        commandPool: u64,
        pAllocator: *const c_void,
    );
    pub(in crate::vulkan) fn vkAllocateCommandBuffers(
        device: *mut c_void,
        pAllocateInfo: *const VkCommandBufferAllocateInfo,
        pCommandBuffers: *mut *mut c_void,
    ) -> i32;
    pub(in crate::vulkan) fn vkBeginCommandBuffer(
        commandBuffer: *mut c_void,
        pBeginInfo: *const VkCommandBufferBeginInfo,
    ) -> i32;
    pub(in crate::vulkan) fn vkEndCommandBuffer(commandBuffer: *mut c_void) -> i32;
    pub(in crate::vulkan) fn vkCmdBeginRenderPass(
        commandBuffer: *mut c_void,
        pRenderPassBegin: *const VkRenderPassBeginInfo,
        contents: i32,
    );
    pub(in crate::vulkan) fn vkCmdEndRenderPass(commandBuffer: *mut c_void);
    pub(in crate::vulkan) fn vkCmdBindPipeline(
        commandBuffer: *mut c_void,
        pipelineBindPoint: i32,
        pipeline: u64,
    );
    pub(in crate::vulkan) fn vkCmdBindVertexBuffers(
        commandBuffer: *mut c_void,
        firstBinding: u32,
        bindingCount: u32,
        pBuffers: *const u64,
        pOffsets: *const u64,
    );
    pub(in crate::vulkan) fn vkCmdDraw(
        commandBuffer: *mut c_void,
        vertexCount: u32,
        instanceCount: u32,
        firstVertex: u32,
        firstInstance: u32,
    );
    pub(in crate::vulkan) fn vkCmdCopyImageToBuffer(
        commandBuffer: *mut c_void,
        srcImage: u64,
        srcImageLayout: i32,
        dstBuffer: u64,
        regionCount: u32,
        pRegions: *const VkBufferImageCopy,
    );
    pub(in crate::vulkan) fn vkCmdPipelineBarrier(
        commandBuffer: *mut c_void,
        srcStageMask: u32,
        dstStageMask: u32,
        dependencyFlags: u32,
        memoryBarrierCount: u32,
        pMemoryBarriers: *const c_void,
        bufferMemoryBarrierCount: u32,
        pBufferMemoryBarriers: *const VkBufferMemoryBarrier,
        imageMemoryBarrierCount: u32,
        pImageMemoryBarriers: *const c_void,
    );
    pub(in crate::vulkan) fn vkGetInstanceProcAddr(
        instance: *mut c_void,
        pName: *const c_char,
    ) -> Option<PFN_vkVoidFunction>;
    pub(in crate::vulkan) fn vkEnumerateInstanceLayerProperties(
        pPropertyCount: *mut u32,
        pProperties: *mut VkLayerProperties,
    ) -> i32;
    pub(in crate::vulkan) fn vkEnumerateInstanceExtensionProperties(
        pLayerName: *const c_char,
        pPropertyCount: *mut u32,
        pProperties: *mut VkExtensionProperties,
    ) -> i32;
    pub(in crate::vulkan) fn vkGetPhysicalDeviceFeatures(
        physicalDevice: *mut c_void,
        pFeatures: *mut VkPhysicalDeviceFeatures,
    );
    pub(in crate::vulkan) fn vkGetFenceStatus(device: *mut c_void, fence: u64) -> i32;
    pub(in crate::vulkan) fn vkCreateSampler(
        device: *mut c_void,
        pCreateInfo: *const VkSamplerCreateInfo,
        pAllocator: *const c_void,
        pSampler: *mut u64,
    ) -> i32;
    pub(in crate::vulkan) fn vkDestroySampler(
        device: *mut c_void,
        sampler: u64,
        pAllocator: *const c_void,
    );
    pub(in crate::vulkan) fn vkCreateDescriptorSetLayout(
        device: *mut c_void,
        pCreateInfo: *const VkDescriptorSetLayoutCreateInfo,
        pAllocator: *const c_void,
        pSetLayout: *mut u64,
    ) -> i32;
    pub(in crate::vulkan) fn vkDestroyDescriptorSetLayout(
        device: *mut c_void,
        descriptorSetLayout: u64,
        pAllocator: *const c_void,
    );
    pub(in crate::vulkan) fn vkCreateDescriptorPool(
        device: *mut c_void,
        pCreateInfo: *const VkDescriptorPoolCreateInfo,
        pAllocator: *const c_void,
        pDescriptorPool: *mut u64,
    ) -> i32;
    pub(in crate::vulkan) fn vkDestroyDescriptorPool(
        device: *mut c_void,
        descriptorPool: u64,
        pAllocator: *const c_void,
    );
    pub(in crate::vulkan) fn vkAllocateDescriptorSets(
        device: *mut c_void,
        pAllocateInfo: *const VkDescriptorSetAllocateInfo,
        pDescriptorSets: *mut u64,
    ) -> i32;
    pub(in crate::vulkan) fn vkFreeDescriptorSets(
        device: *mut c_void,
        descriptorPool: u64,
        descriptorSetCount: u32,
        pDescriptorSets: *const u64,
    ) -> i32;
    pub(in crate::vulkan) fn vkUpdateDescriptorSets(
        device: *mut c_void,
        descriptorWriteCount: u32,
        pDescriptorWrites: *const VkWriteDescriptorSet,
        descriptorCopyCount: u32,
        pDescriptorCopies: *const c_void,
    );
    pub(in crate::vulkan) fn vkCreateComputePipelines(
        device: *mut c_void,
        pipelineCache: u64,
        createInfoCount: u32,
        pCreateInfos: *const VkComputePipelineCreateInfo,
        pAllocator: *const c_void,
        pPipelines: *mut u64,
    ) -> i32;
    pub(in crate::vulkan) fn vkCmdBindDescriptorSets(
        commandBuffer: *mut c_void,
        pipelineBindPoint: i32,
        layout: u64,
        firstSet: u32,
        descriptorSetCount: u32,
        pDescriptorSets: *const u64,
        dynamicOffsetCount: u32,
        pDynamicOffsets: *const u32,
    );
    pub(in crate::vulkan) fn vkCmdBindIndexBuffer(
        commandBuffer: *mut c_void,
        buffer: u64,
        offset: u64,
        indexType: i32,
    );
    pub(in crate::vulkan) fn vkCmdDrawIndexed(
        commandBuffer: *mut c_void,
        indexCount: u32,
        instanceCount: u32,
        firstIndex: u32,
        vertexOffset: i32,
        firstInstance: u32,
    );
    pub(in crate::vulkan) fn vkCmdDispatch(
        commandBuffer: *mut c_void,
        groupCountX: u32,
        groupCountY: u32,
        groupCountZ: u32,
    );
    pub(in crate::vulkan) fn vkCmdSetViewport(
        commandBuffer: *mut c_void,
        firstViewport: u32,
        viewportCount: u32,
        pViewports: *const VkViewport,
    );
    pub(in crate::vulkan) fn vkCmdSetScissor(
        commandBuffer: *mut c_void,
        firstScissor: u32,
        scissorCount: u32,
        pScissors: *const VkRect2D,
    );
    pub(in crate::vulkan) fn vkCmdCopyBufferToImage(
        commandBuffer: *mut c_void,
        srcBuffer: u64,
        dstImage: u64,
        dstImageLayout: i32,
        regionCount: u32,
        pRegions: *const VkBufferImageCopy,
    );
    pub(in crate::vulkan) fn vkCmdBlitImage(
        commandBuffer: *mut c_void,
        srcImage: u64,
        srcImageLayout: i32,
        dstImage: u64,
        dstImageLayout: i32,
        regionCount: u32,
        pRegions: *const VkImageBlit,
        filter: i32,
    );
}
