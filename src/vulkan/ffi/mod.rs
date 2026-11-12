// Vulkan registry declarations: Copyright 2015-2024 The Khronos Group Inc.
// Adapted from Vulkan-Headers v1.3.275 under Apache-2.0. See docs/Vulkan-Registry-LICENSE.txt.
//! Minimal Vulkan 1.0 C ABI declarations, derived from Khronos Vulkan-Headers v1.3.275.
#![allow(non_snake_case, non_camel_case_types, dead_code)]
use core::ffi::{c_char, c_void};

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct VkXlibSurfaceCreateInfoKHR {
    pub sType: i32,
    pub pNext: *const c_void,
    pub flags: u32,
    pub dpy: *mut c_void,
    pub window: core::ffi::c_ulong,
}
impl VkXlibSurfaceCreateInfoKHR {
    pub fn new() -> Self {
        // All other fields are C scalars, handles or pointers; callers populate required fields.
        Self {
            sType: 1000004000,
            ..unsafe { core::mem::zeroed() }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct VkWin32SurfaceCreateInfoKHR {
    pub sType: i32,
    pub pNext: *const c_void,
    pub flags: u32,
    pub hinstance: *mut c_void,
    pub hwnd: *mut c_void,
}
impl VkWin32SurfaceCreateInfoKHR {
    pub fn new() -> Self {
        // All other fields are C scalars, handles or pointers; callers populate required fields.
        Self {
            sType: 1000009000,
            ..unsafe { core::mem::zeroed() }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct VkExtent2D {
    pub width: u32,
    pub height: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct VkSurfaceCapabilitiesKHR {
    pub minImageCount: u32,
    pub maxImageCount: u32,
    pub currentExtent: VkExtent2D,
    pub minImageExtent: VkExtent2D,
    pub maxImageExtent: VkExtent2D,
    pub maxImageArrayLayers: u32,
    pub supportedTransforms: u32,
    pub currentTransform: i32,
    pub supportedCompositeAlpha: u32,
    pub supportedUsageFlags: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct VkSurfaceFormatKHR {
    pub format: i32,
    pub colorSpace: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct VkSwapchainCreateInfoKHR {
    pub sType: i32,
    pub pNext: *const c_void,
    pub flags: u32,
    pub surface: u64,
    pub minImageCount: u32,
    pub imageFormat: i32,
    pub imageColorSpace: i32,
    pub imageExtent: VkExtent2D,
    pub imageArrayLayers: u32,
    pub imageUsage: u32,
    pub imageSharingMode: i32,
    pub queueFamilyIndexCount: u32,
    pub pQueueFamilyIndices: *const u32,
    pub preTransform: i32,
    pub compositeAlpha: i32,
    pub presentMode: i32,
    pub clipped: u32,
    pub oldSwapchain: u64,
}
impl VkSwapchainCreateInfoKHR {
    pub fn new() -> Self {
        // All other fields are C scalars, handles or pointers; callers populate required fields.
        Self {
            sType: 1000001000,
            ..unsafe { core::mem::zeroed() }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct VkPresentInfoKHR {
    pub sType: i32,
    pub pNext: *const c_void,
    pub waitSemaphoreCount: u32,
    pub pWaitSemaphores: *const u64,
    pub swapchainCount: u32,
    pub pSwapchains: *const u64,
    pub pImageIndices: *const u32,
    pub pResults: *mut i32,
}
impl VkPresentInfoKHR {
    pub fn new() -> Self {
        // All other fields are C scalars, handles or pointers; callers populate required fields.
        Self {
            sType: 1000001001,
            ..unsafe { core::mem::zeroed() }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct VkFenceCreateInfo {
    pub sType: i32,
    pub pNext: *const c_void,
    pub flags: u32,
}
impl VkFenceCreateInfo {
    pub fn new() -> Self {
        // All other fields are C scalars, handles or pointers; callers populate required fields.
        Self {
            sType: 8,
            ..unsafe { core::mem::zeroed() }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct VkSemaphoreCreateInfo {
    pub sType: i32,
    pub pNext: *const c_void,
    pub flags: u32,
}
impl VkSemaphoreCreateInfo {
    pub fn new() -> Self {
        // All other fields are C scalars, handles or pointers; callers populate required fields.
        Self {
            sType: 9,
            ..unsafe { core::mem::zeroed() }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct VkBufferCopy {
    pub srcOffset: u64,
    pub dstOffset: u64,
    pub size: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct VkQueryPoolCreateInfo {
    pub sType: i32,
    pub pNext: *const c_void,
    pub flags: u32,
    pub queryType: i32,
    pub queryCount: u32,
    pub pipelineStatistics: u32,
}
impl VkQueryPoolCreateInfo {
    pub fn new() -> Self {
        // All other fields are C scalars, handles or pointers; callers populate required fields.
        Self {
            sType: 11,
            ..unsafe { core::mem::zeroed() }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct VkApplicationInfo {
    pub sType: i32,
    pub pNext: *const c_void,
    pub pApplicationName: *const c_char,
    pub applicationVersion: u32,
    pub pEngineName: *const c_char,
    pub engineVersion: u32,
    pub apiVersion: u32,
}
impl VkApplicationInfo {
    pub fn new() -> Self {
        // All other fields are C scalars, handles or pointers; callers populate required fields.
        Self {
            sType: 0,
            ..unsafe { core::mem::zeroed() }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct VkInstanceCreateInfo {
    pub sType: i32,
    pub pNext: *const c_void,
    pub flags: u32,
    pub pApplicationInfo: *const VkApplicationInfo,
    pub enabledLayerCount: u32,
    pub ppEnabledLayerNames: *const *const c_char,
    pub enabledExtensionCount: u32,
    pub ppEnabledExtensionNames: *const *const c_char,
}
impl VkInstanceCreateInfo {
    pub fn new() -> Self {
        // All other fields are C scalars, handles or pointers; callers populate required fields.
        Self {
            sType: 1,
            ..unsafe { core::mem::zeroed() }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct VkExtent3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct VkQueueFamilyProperties {
    pub queueFlags: u32,
    pub queueCount: u32,
    pub timestampValidBits: u32,
    pub minImageTransferGranularity: VkExtent3D,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct VkMemoryType {
    pub propertyFlags: u32,
    pub heapIndex: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct VkMemoryHeap {
    pub size: u64,
    pub flags: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct VkPhysicalDeviceMemoryProperties {
    pub memoryTypeCount: u32,
    pub memoryTypes: [VkMemoryType; 32],
    pub memoryHeapCount: u32,
    pub memoryHeaps: [VkMemoryHeap; 16],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct VkPhysicalDeviceLimits {
    pub maxImageDimension1D: u32,
    pub maxImageDimension2D: u32,
    pub maxImageDimension3D: u32,
    pub maxImageDimensionCube: u32,
    pub maxImageArrayLayers: u32,
    pub maxTexelBufferElements: u32,
    pub maxUniformBufferRange: u32,
    pub maxStorageBufferRange: u32,
    pub maxPushConstantsSize: u32,
    pub maxMemoryAllocationCount: u32,
    pub maxSamplerAllocationCount: u32,
    pub bufferImageGranularity: u64,
    pub sparseAddressSpaceSize: u64,
    pub maxBoundDescriptorSets: u32,
    pub maxPerStageDescriptorSamplers: u32,
    pub maxPerStageDescriptorUniformBuffers: u32,
    pub maxPerStageDescriptorStorageBuffers: u32,
    pub maxPerStageDescriptorSampledImages: u32,
    pub maxPerStageDescriptorStorageImages: u32,
    pub maxPerStageDescriptorInputAttachments: u32,
    pub maxPerStageResources: u32,
    pub maxDescriptorSetSamplers: u32,
    pub maxDescriptorSetUniformBuffers: u32,
    pub maxDescriptorSetUniformBuffersDynamic: u32,
    pub maxDescriptorSetStorageBuffers: u32,
    pub maxDescriptorSetStorageBuffersDynamic: u32,
    pub maxDescriptorSetSampledImages: u32,
    pub maxDescriptorSetStorageImages: u32,
    pub maxDescriptorSetInputAttachments: u32,
    pub maxVertexInputAttributes: u32,
    pub maxVertexInputBindings: u32,
    pub maxVertexInputAttributeOffset: u32,
    pub maxVertexInputBindingStride: u32,
    pub maxVertexOutputComponents: u32,
    pub maxTessellationGenerationLevel: u32,
    pub maxTessellationPatchSize: u32,
    pub maxTessellationControlPerVertexInputComponents: u32,
    pub maxTessellationControlPerVertexOutputComponents: u32,
    pub maxTessellationControlPerPatchOutputComponents: u32,
    pub maxTessellationControlTotalOutputComponents: u32,
    pub maxTessellationEvaluationInputComponents: u32,
    pub maxTessellationEvaluationOutputComponents: u32,
    pub maxGeometryShaderInvocations: u32,
    pub maxGeometryInputComponents: u32,
    pub maxGeometryOutputComponents: u32,
    pub maxGeometryOutputVertices: u32,
    pub maxGeometryTotalOutputComponents: u32,
    pub maxFragmentInputComponents: u32,
    pub maxFragmentOutputAttachments: u32,
    pub maxFragmentDualSrcAttachments: u32,
    pub maxFragmentCombinedOutputResources: u32,
    pub maxComputeSharedMemorySize: u32,
    pub maxComputeWorkGroupCount: [u32; 3],
    pub maxComputeWorkGroupInvocations: u32,
    pub maxComputeWorkGroupSize: [u32; 3],
    pub subPixelPrecisionBits: u32,
    pub subTexelPrecisionBits: u32,
    pub mipmapPrecisionBits: u32,
    pub maxDrawIndexedIndexValue: u32,
    pub maxDrawIndirectCount: u32,
    pub maxSamplerLodBias: f32,
    pub maxSamplerAnisotropy: f32,
    pub maxViewports: u32,
    pub maxViewportDimensions: [u32; 2],
    pub viewportBoundsRange: [f32; 2],
    pub viewportSubPixelBits: u32,
    pub minMemoryMapAlignment: usize,
    pub minTexelBufferOffsetAlignment: u64,
    pub minUniformBufferOffsetAlignment: u64,
    pub minStorageBufferOffsetAlignment: u64,
    pub minTexelOffset: i32,
    pub maxTexelOffset: u32,
    pub minTexelGatherOffset: i32,
    pub maxTexelGatherOffset: u32,
    pub minInterpolationOffset: f32,
    pub maxInterpolationOffset: f32,
    pub subPixelInterpolationOffsetBits: u32,
    pub maxFramebufferWidth: u32,
    pub maxFramebufferHeight: u32,
    pub maxFramebufferLayers: u32,
    pub framebufferColorSampleCounts: u32,
    pub framebufferDepthSampleCounts: u32,
    pub framebufferStencilSampleCounts: u32,
    pub framebufferNoAttachmentsSampleCounts: u32,
    pub maxColorAttachments: u32,
    pub sampledImageColorSampleCounts: u32,
    pub sampledImageIntegerSampleCounts: u32,
    pub sampledImageDepthSampleCounts: u32,
    pub sampledImageStencilSampleCounts: u32,
    pub storageImageSampleCounts: u32,
    pub maxSampleMaskWords: u32,
    pub timestampComputeAndGraphics: u32,
    pub timestampPeriod: f32,
    pub maxClipDistances: u32,
    pub maxCullDistances: u32,
    pub maxCombinedClipAndCullDistances: u32,
    pub discreteQueuePriorities: u32,
    pub pointSizeRange: [f32; 2],
    pub lineWidthRange: [f32; 2],
    pub pointSizeGranularity: f32,
    pub lineWidthGranularity: f32,
    pub strictLines: u32,
    pub standardSampleLocations: u32,
    pub optimalBufferCopyOffsetAlignment: u64,
    pub optimalBufferCopyRowPitchAlignment: u64,
    pub nonCoherentAtomSize: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct VkPhysicalDeviceSparseProperties {
    pub residencyStandard2DBlockShape: u32,
    pub residencyStandard2DMultisampleBlockShape: u32,
    pub residencyStandard3DBlockShape: u32,
    pub residencyAlignedMipSize: u32,
    pub residencyNonResidentStrict: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct VkPhysicalDeviceProperties {
    pub apiVersion: u32,
    pub driverVersion: u32,
    pub vendorID: u32,
    pub deviceID: u32,
    pub deviceType: i32,
    pub deviceName: [c_char; 256],
    pub pipelineCacheUUID: [u8; 16],
    pub limits: VkPhysicalDeviceLimits,
    pub sparseProperties: VkPhysicalDeviceSparseProperties,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct VkFormatProperties {
    pub linearTilingFeatures: u32,
    pub optimalTilingFeatures: u32,
    pub bufferFeatures: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct VkDeviceQueueCreateInfo {
    pub sType: i32,
    pub pNext: *const c_void,
    pub flags: u32,
    pub queueFamilyIndex: u32,
    pub queueCount: u32,
    pub pQueuePriorities: *const f32,
}
impl VkDeviceQueueCreateInfo {
    pub fn new() -> Self {
        // All other fields are C scalars, handles or pointers; callers populate required fields.
        Self {
            sType: 2,
            ..unsafe { core::mem::zeroed() }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct VkDeviceCreateInfo {
    pub sType: i32,
    pub pNext: *const c_void,
    pub flags: u32,
    pub queueCreateInfoCount: u32,
    pub pQueueCreateInfos: *const VkDeviceQueueCreateInfo,
    pub enabledLayerCount: u32,
    pub ppEnabledLayerNames: *const *const c_char,
    pub enabledExtensionCount: u32,
    pub ppEnabledExtensionNames: *const *const c_char,
    pub pEnabledFeatures: *const c_void,
}
impl VkDeviceCreateInfo {
    pub fn new() -> Self {
        // All other fields are C scalars, handles or pointers; callers populate required fields.
        Self {
            sType: 3,
            ..unsafe { core::mem::zeroed() }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct VkSubmitInfo {
    pub sType: i32,
    pub pNext: *const c_void,
    pub waitSemaphoreCount: u32,
    pub pWaitSemaphores: *const u64,
    pub pWaitDstStageMask: *const u32,
    pub commandBufferCount: u32,
    pub pCommandBuffers: *const *mut c_void,
    pub signalSemaphoreCount: u32,
    pub pSignalSemaphores: *const u64,
}
impl VkSubmitInfo {
    pub fn new() -> Self {
        // All other fields are C scalars, handles or pointers; callers populate required fields.
        Self {
            sType: 4,
            ..unsafe { core::mem::zeroed() }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct VkBufferCreateInfo {
    pub sType: i32,
    pub pNext: *const c_void,
    pub flags: u32,
    pub size: u64,
    pub usage: u32,
    pub sharingMode: i32,
    pub queueFamilyIndexCount: u32,
    pub pQueueFamilyIndices: *const u32,
}
impl VkBufferCreateInfo {
    pub fn new() -> Self {
        // All other fields are C scalars, handles or pointers; callers populate required fields.
        Self {
            sType: 12,
            ..unsafe { core::mem::zeroed() }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct VkMemoryRequirements {
    pub size: u64,
    pub alignment: u64,
    pub memoryTypeBits: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct VkImageCreateInfo {
    pub sType: i32,
    pub pNext: *const c_void,
    pub flags: u32,
    pub imageType: i32,
    pub format: i32,
    pub extent: VkExtent3D,
    pub mipLevels: u32,
    pub arrayLayers: u32,
    pub samples: i32,
    pub tiling: i32,
    pub usage: u32,
    pub sharingMode: i32,
    pub queueFamilyIndexCount: u32,
    pub pQueueFamilyIndices: *const u32,
    pub initialLayout: i32,
}
impl VkImageCreateInfo {
    pub fn new() -> Self {
        // All other fields are C scalars, handles or pointers; callers populate required fields.
        Self {
            sType: 14,
            ..unsafe { core::mem::zeroed() }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct VkMemoryAllocateInfo {
    pub sType: i32,
    pub pNext: *const c_void,
    pub allocationSize: u64,
    pub memoryTypeIndex: u32,
}
impl VkMemoryAllocateInfo {
    pub fn new() -> Self {
        // All other fields are C scalars, handles or pointers; callers populate required fields.
        Self {
            sType: 5,
            ..unsafe { core::mem::zeroed() }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct VkComponentMapping {
    pub r: i32,
    pub g: i32,
    pub b: i32,
    pub a: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct VkImageSubresourceRange {
    pub aspectMask: u32,
    pub baseMipLevel: u32,
    pub levelCount: u32,
    pub baseArrayLayer: u32,
    pub layerCount: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct VkImageViewCreateInfo {
    pub sType: i32,
    pub pNext: *const c_void,
    pub flags: u32,
    pub image: u64,
    pub viewType: i32,
    pub format: i32,
    pub components: VkComponentMapping,
    pub subresourceRange: VkImageSubresourceRange,
}
impl VkImageViewCreateInfo {
    pub fn new() -> Self {
        // All other fields are C scalars, handles or pointers; callers populate required fields.
        Self {
            sType: 15,
            ..unsafe { core::mem::zeroed() }
        }
    }
}

mod commands;
mod types;

pub(super) use commands::*;
pub(super) use types::*;
