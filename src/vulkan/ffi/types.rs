use core::ffi::{c_char, c_void};

use super::{VkExtent2D, VkExtent3D, VkImageSubresourceRange};

#[repr(C)]
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct VkAttachmentDescription {
    pub flags: u32,
    pub format: i32,
    pub samples: i32,
    pub loadOp: i32,
    pub storeOp: i32,
    pub stencilLoadOp: i32,
    pub stencilStoreOp: i32,
    pub initialLayout: i32,
    pub finalLayout: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct VkAttachmentReference {
    pub attachment: u32,
    pub layout: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct VkSubpassDescription {
    pub flags: u32,
    pub pipelineBindPoint: i32,
    pub inputAttachmentCount: u32,
    pub pInputAttachments: *const VkAttachmentReference,
    pub colorAttachmentCount: u32,
    pub pColorAttachments: *const VkAttachmentReference,
    pub pResolveAttachments: *const VkAttachmentReference,
    pub pDepthStencilAttachment: *const VkAttachmentReference,
    pub preserveAttachmentCount: u32,
    pub pPreserveAttachments: *const u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct VkSubpassDependency {
    pub srcSubpass: u32,
    pub dstSubpass: u32,
    pub srcStageMask: u32,
    pub dstStageMask: u32,
    pub srcAccessMask: u32,
    pub dstAccessMask: u32,
    pub dependencyFlags: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct VkRenderPassCreateInfo {
    pub sType: i32,
    pub pNext: *const c_void,
    pub flags: u32,
    pub attachmentCount: u32,
    pub pAttachments: *const VkAttachmentDescription,
    pub subpassCount: u32,
    pub pSubpasses: *const VkSubpassDescription,
    pub dependencyCount: u32,
    pub pDependencies: *const VkSubpassDependency,
}
impl VkRenderPassCreateInfo {
    pub fn new() -> Self {
        // All other fields are C scalars, handles or pointers; callers populate required fields.
        Self {
            sType: 38,
            ..unsafe { core::mem::zeroed() }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct VkFramebufferCreateInfo {
    pub sType: i32,
    pub pNext: *const c_void,
    pub flags: u32,
    pub renderPass: u64,
    pub attachmentCount: u32,
    pub pAttachments: *const u64,
    pub width: u32,
    pub height: u32,
    pub layers: u32,
}
impl VkFramebufferCreateInfo {
    pub fn new() -> Self {
        // All other fields are C scalars, handles or pointers; callers populate required fields.
        Self {
            sType: 37,
            ..unsafe { core::mem::zeroed() }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct VkShaderModuleCreateInfo {
    pub sType: i32,
    pub pNext: *const c_void,
    pub flags: u32,
    pub codeSize: usize,
    pub pCode: *const u32,
}
impl VkShaderModuleCreateInfo {
    pub fn new() -> Self {
        // All other fields are C scalars, handles or pointers; callers populate required fields.
        Self {
            sType: 16,
            ..unsafe { core::mem::zeroed() }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct VkPipelineCacheCreateInfo {
    pub sType: i32,
    pub pNext: *const c_void,
    pub flags: u32,
    pub initialDataSize: usize,
    pub pInitialData: *const c_void,
}
impl VkPipelineCacheCreateInfo {
    pub fn new() -> Self {
        Self {
            sType: 17,
            ..unsafe { core::mem::zeroed() }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct VkPushConstantRange {
    pub stageFlags: u32,
    pub offset: u32,
    pub size: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct VkPipelineLayoutCreateInfo {
    pub sType: i32,
    pub pNext: *const c_void,
    pub flags: u32,
    pub setLayoutCount: u32,
    pub pSetLayouts: *const u64,
    pub pushConstantRangeCount: u32,
    pub pPushConstantRanges: *const VkPushConstantRange,
}
impl VkPipelineLayoutCreateInfo {
    pub fn new() -> Self {
        // All other fields are C scalars, handles or pointers; callers populate required fields.
        Self {
            sType: 30,
            ..unsafe { core::mem::zeroed() }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct VkPipelineShaderStageCreateInfo {
    pub sType: i32,
    pub pNext: *const c_void,
    pub flags: u32,
    pub stage: i32,
    pub module: u64,
    pub pName: *const c_char,
    pub pSpecializationInfo: *const c_void,
}
impl VkPipelineShaderStageCreateInfo {
    pub fn new() -> Self {
        // All other fields are C scalars, handles or pointers; callers populate required fields.
        Self {
            sType: 18,
            ..unsafe { core::mem::zeroed() }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct VkVertexInputBindingDescription {
    pub binding: u32,
    pub stride: u32,
    pub inputRate: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct VkVertexInputAttributeDescription {
    pub location: u32,
    pub binding: u32,
    pub format: i32,
    pub offset: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct VkPipelineVertexInputStateCreateInfo {
    pub sType: i32,
    pub pNext: *const c_void,
    pub flags: u32,
    pub vertexBindingDescriptionCount: u32,
    pub pVertexBindingDescriptions: *const VkVertexInputBindingDescription,
    pub vertexAttributeDescriptionCount: u32,
    pub pVertexAttributeDescriptions: *const VkVertexInputAttributeDescription,
}
impl VkPipelineVertexInputStateCreateInfo {
    pub fn new() -> Self {
        // All other fields are C scalars, handles or pointers; callers populate required fields.
        Self {
            sType: 19,
            ..unsafe { core::mem::zeroed() }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct VkPipelineInputAssemblyStateCreateInfo {
    pub sType: i32,
    pub pNext: *const c_void,
    pub flags: u32,
    pub topology: i32,
    pub primitiveRestartEnable: u32,
}
impl VkPipelineInputAssemblyStateCreateInfo {
    pub fn new() -> Self {
        // All other fields are C scalars, handles or pointers; callers populate required fields.
        Self {
            sType: 20,
            ..unsafe { core::mem::zeroed() }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct VkViewport {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub minDepth: f32,
    pub maxDepth: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct VkOffset2D {
    pub x: i32,
    pub y: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct VkRect2D {
    pub offset: VkOffset2D,
    pub extent: VkExtent2D,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct VkPipelineViewportStateCreateInfo {
    pub sType: i32,
    pub pNext: *const c_void,
    pub flags: u32,
    pub viewportCount: u32,
    pub pViewports: *const VkViewport,
    pub scissorCount: u32,
    pub pScissors: *const VkRect2D,
}
impl VkPipelineViewportStateCreateInfo {
    pub fn new() -> Self {
        // All other fields are C scalars, handles or pointers; callers populate required fields.
        Self {
            sType: 22,
            ..unsafe { core::mem::zeroed() }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct VkPipelineRasterizationStateCreateInfo {
    pub sType: i32,
    pub pNext: *const c_void,
    pub flags: u32,
    pub depthClampEnable: u32,
    pub rasterizerDiscardEnable: u32,
    pub polygonMode: i32,
    pub cullMode: u32,
    pub frontFace: i32,
    pub depthBiasEnable: u32,
    pub depthBiasConstantFactor: f32,
    pub depthBiasClamp: f32,
    pub depthBiasSlopeFactor: f32,
    pub lineWidth: f32,
}
impl VkPipelineRasterizationStateCreateInfo {
    pub fn new() -> Self {
        // All other fields are C scalars, handles or pointers; callers populate required fields.
        Self {
            sType: 23,
            ..unsafe { core::mem::zeroed() }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct VkPipelineMultisampleStateCreateInfo {
    pub sType: i32,
    pub pNext: *const c_void,
    pub flags: u32,
    pub rasterizationSamples: i32,
    pub sampleShadingEnable: u32,
    pub minSampleShading: f32,
    pub pSampleMask: *const u32,
    pub alphaToCoverageEnable: u32,
    pub alphaToOneEnable: u32,
}
impl VkPipelineMultisampleStateCreateInfo {
    pub fn new() -> Self {
        // All other fields are C scalars, handles or pointers; callers populate required fields.
        Self {
            sType: 24,
            ..unsafe { core::mem::zeroed() }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct VkStencilOpState {
    pub failOp: i32,
    pub passOp: i32,
    pub depthFailOp: i32,
    pub compareOp: i32,
    pub compareMask: u32,
    pub writeMask: u32,
    pub reference: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct VkPipelineDepthStencilStateCreateInfo {
    pub sType: i32,
    pub pNext: *const c_void,
    pub flags: u32,
    pub depthTestEnable: u32,
    pub depthWriteEnable: u32,
    pub depthCompareOp: i32,
    pub depthBoundsTestEnable: u32,
    pub stencilTestEnable: u32,
    pub front: VkStencilOpState,
    pub back: VkStencilOpState,
    pub minDepthBounds: f32,
    pub maxDepthBounds: f32,
}
impl VkPipelineDepthStencilStateCreateInfo {
    pub fn new() -> Self {
        // All other fields are C scalars, handles or pointers; callers populate required fields.
        Self {
            sType: 25,
            ..unsafe { core::mem::zeroed() }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct VkPipelineColorBlendAttachmentState {
    pub blendEnable: u32,
    pub srcColorBlendFactor: i32,
    pub dstColorBlendFactor: i32,
    pub colorBlendOp: i32,
    pub srcAlphaBlendFactor: i32,
    pub dstAlphaBlendFactor: i32,
    pub alphaBlendOp: i32,
    pub colorWriteMask: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct VkPipelineColorBlendStateCreateInfo {
    pub sType: i32,
    pub pNext: *const c_void,
    pub flags: u32,
    pub logicOpEnable: u32,
    pub logicOp: i32,
    pub attachmentCount: u32,
    pub pAttachments: *const VkPipelineColorBlendAttachmentState,
    pub blendConstants: [f32; 4],
}
impl VkPipelineColorBlendStateCreateInfo {
    pub fn new() -> Self {
        // All other fields are C scalars, handles or pointers; callers populate required fields.
        Self {
            sType: 26,
            ..unsafe { core::mem::zeroed() }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct VkGraphicsPipelineCreateInfo {
    pub sType: i32,
    pub pNext: *const c_void,
    pub flags: u32,
    pub stageCount: u32,
    pub pStages: *const VkPipelineShaderStageCreateInfo,
    pub pVertexInputState: *const VkPipelineVertexInputStateCreateInfo,
    pub pInputAssemblyState: *const VkPipelineInputAssemblyStateCreateInfo,
    pub pTessellationState: *const c_void,
    pub pViewportState: *const VkPipelineViewportStateCreateInfo,
    pub pRasterizationState: *const VkPipelineRasterizationStateCreateInfo,
    pub pMultisampleState: *const VkPipelineMultisampleStateCreateInfo,
    pub pDepthStencilState: *const VkPipelineDepthStencilStateCreateInfo,
    pub pColorBlendState: *const VkPipelineColorBlendStateCreateInfo,
    pub pDynamicState: *const c_void,
    pub layout: u64,
    pub renderPass: u64,
    pub subpass: u32,
    pub basePipelineHandle: u64,
    pub basePipelineIndex: i32,
}
impl VkGraphicsPipelineCreateInfo {
    pub fn new() -> Self {
        // All other fields are C scalars, handles or pointers; callers populate required fields.
        Self {
            sType: 28,
            ..unsafe { core::mem::zeroed() }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct VkCommandPoolCreateInfo {
    pub sType: i32,
    pub pNext: *const c_void,
    pub flags: u32,
    pub queueFamilyIndex: u32,
}
impl VkCommandPoolCreateInfo {
    pub fn new() -> Self {
        // All other fields are C scalars, handles or pointers; callers populate required fields.
        Self {
            sType: 39,
            ..unsafe { core::mem::zeroed() }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct VkCommandBufferAllocateInfo {
    pub sType: i32,
    pub pNext: *const c_void,
    pub commandPool: u64,
    pub level: i32,
    pub commandBufferCount: u32,
}
impl VkCommandBufferAllocateInfo {
    pub fn new() -> Self {
        // All other fields are C scalars, handles or pointers; callers populate required fields.
        Self {
            sType: 40,
            ..unsafe { core::mem::zeroed() }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct VkCommandBufferInheritanceInfo {
    pub sType: i32,
    pub pNext: *const c_void,
    pub renderPass: u64,
    pub subpass: u32,
    pub framebuffer: u64,
    pub occlusionQueryEnable: u32,
    pub queryFlags: u32,
    pub pipelineStatistics: u32,
}
impl VkCommandBufferInheritanceInfo {
    pub fn new() -> Self {
        // All other fields are C scalars, handles or pointers; callers populate required fields.
        Self {
            sType: 41,
            ..unsafe { core::mem::zeroed() }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct VkCommandBufferBeginInfo {
    pub sType: i32,
    pub pNext: *const c_void,
    pub flags: u32,
    pub pInheritanceInfo: *const VkCommandBufferInheritanceInfo,
}
impl VkCommandBufferBeginInfo {
    pub fn new() -> Self {
        // All other fields are C scalars, handles or pointers; callers populate required fields.
        Self {
            sType: 42,
            ..unsafe { core::mem::zeroed() }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(in crate::vulkan) union VkClearColorValue {
    pub float32: [f32; 4],
    pub int32: [i32; 4],
    pub uint32: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct VkClearDepthStencilValue {
    pub depth: f32,
    pub stencil: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(in crate::vulkan) union VkClearValue {
    pub color: VkClearColorValue,
    pub depthStencil: VkClearDepthStencilValue,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct VkRenderPassBeginInfo {
    pub sType: i32,
    pub pNext: *const c_void,
    pub renderPass: u64,
    pub framebuffer: u64,
    pub renderArea: VkRect2D,
    pub clearValueCount: u32,
    pub pClearValues: *const VkClearValue,
}
impl VkRenderPassBeginInfo {
    pub fn new() -> Self {
        // All other fields are C scalars, handles or pointers; callers populate required fields.
        Self {
            sType: 43,
            ..unsafe { core::mem::zeroed() }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct VkImageSubresourceLayers {
    pub aspectMask: u32,
    pub mipLevel: u32,
    pub baseArrayLayer: u32,
    pub layerCount: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct VkOffset3D {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct VkBufferImageCopy {
    pub bufferOffset: u64,
    pub bufferRowLength: u32,
    pub bufferImageHeight: u32,
    pub imageSubresource: VkImageSubresourceLayers,
    pub imageOffset: VkOffset3D,
    pub imageExtent: VkExtent3D,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct VkBufferMemoryBarrier {
    pub sType: i32,
    pub pNext: *const c_void,
    pub srcAccessMask: u32,
    pub dstAccessMask: u32,
    pub srcQueueFamilyIndex: u32,
    pub dstQueueFamilyIndex: u32,
    pub buffer: u64,
    pub offset: u64,
    pub size: u64,
}
impl VkBufferMemoryBarrier {
    pub fn new() -> Self {
        // All other fields are C scalars, handles or pointers; callers populate required fields.
        Self {
            sType: 44,
            ..unsafe { core::mem::zeroed() }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct VkPhysicalDeviceFeatures {
    pub robustBufferAccess: u32,
    pub fullDrawIndexUint32: u32,
    pub imageCubeArray: u32,
    pub independentBlend: u32,
    pub geometryShader: u32,
    pub tessellationShader: u32,
    pub sampleRateShading: u32,
    pub dualSrcBlend: u32,
    pub logicOp: u32,
    pub multiDrawIndirect: u32,
    pub drawIndirectFirstInstance: u32,
    pub depthClamp: u32,
    pub depthBiasClamp: u32,
    pub fillModeNonSolid: u32,
    pub depthBounds: u32,
    pub wideLines: u32,
    pub largePoints: u32,
    pub alphaToOne: u32,
    pub multiViewport: u32,
    pub samplerAnisotropy: u32,
    pub textureCompressionETC2: u32,
    pub textureCompressionASTC_LDR: u32,
    pub textureCompressionBC: u32,
    pub occlusionQueryPrecise: u32,
    pub pipelineStatisticsQuery: u32,
    pub vertexPipelineStoresAndAtomics: u32,
    pub fragmentStoresAndAtomics: u32,
    pub shaderTessellationAndGeometryPointSize: u32,
    pub shaderImageGatherExtended: u32,
    pub shaderStorageImageExtendedFormats: u32,
    pub shaderStorageImageMultisample: u32,
    pub shaderStorageImageReadWithoutFormat: u32,
    pub shaderStorageImageWriteWithoutFormat: u32,
    pub shaderUniformBufferArrayDynamicIndexing: u32,
    pub shaderSampledImageArrayDynamicIndexing: u32,
    pub shaderStorageBufferArrayDynamicIndexing: u32,
    pub shaderStorageImageArrayDynamicIndexing: u32,
    pub shaderClipDistance: u32,
    pub shaderCullDistance: u32,
    pub shaderFloat64: u32,
    pub shaderInt64: u32,
    pub shaderInt16: u32,
    pub shaderResourceResidency: u32,
    pub shaderResourceMinLod: u32,
    pub sparseBinding: u32,
    pub sparseResidencyBuffer: u32,
    pub sparseResidencyImage2D: u32,
    pub sparseResidencyImage3D: u32,
    pub sparseResidency2Samples: u32,
    pub sparseResidency4Samples: u32,
    pub sparseResidency8Samples: u32,
    pub sparseResidency16Samples: u32,
    pub sparseResidencyAliased: u32,
    pub variableMultisampleRate: u32,
    pub inheritedQueries: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct VkMemoryBarrier {
    pub sType: i32,
    pub pNext: *const c_void,
    pub srcAccessMask: u32,
    pub dstAccessMask: u32,
}
impl VkMemoryBarrier {
    pub fn new() -> Self {
        // All other fields are C scalars, handles or pointers; callers populate required fields.
        Self {
            sType: 46,
            ..unsafe { core::mem::zeroed() }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct VkImageMemoryBarrier {
    pub sType: i32,
    pub pNext: *const c_void,
    pub srcAccessMask: u32,
    pub dstAccessMask: u32,
    pub oldLayout: i32,
    pub newLayout: i32,
    pub srcQueueFamilyIndex: u32,
    pub dstQueueFamilyIndex: u32,
    pub image: u64,
    pub subresourceRange: VkImageSubresourceRange,
}
impl VkImageMemoryBarrier {
    pub fn new() -> Self {
        // All other fields are C scalars, handles or pointers; callers populate required fields.
        Self {
            sType: 45,
            ..unsafe { core::mem::zeroed() }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct VkSamplerCreateInfo {
    pub sType: i32,
    pub pNext: *const c_void,
    pub flags: u32,
    pub magFilter: i32,
    pub minFilter: i32,
    pub mipmapMode: i32,
    pub addressModeU: i32,
    pub addressModeV: i32,
    pub addressModeW: i32,
    pub mipLodBias: f32,
    pub anisotropyEnable: u32,
    pub maxAnisotropy: f32,
    pub compareEnable: u32,
    pub compareOp: i32,
    pub minLod: f32,
    pub maxLod: f32,
    pub borderColor: i32,
    pub unnormalizedCoordinates: u32,
}
impl VkSamplerCreateInfo {
    pub fn new() -> Self {
        // All other fields are C scalars, handles or pointers; callers populate required fields.
        Self {
            sType: 31,
            ..unsafe { core::mem::zeroed() }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct VkDescriptorSetLayoutBinding {
    pub binding: u32,
    pub descriptorType: i32,
    pub descriptorCount: u32,
    pub stageFlags: u32,
    pub pImmutableSamplers: *const u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct VkDescriptorSetLayoutCreateInfo {
    pub sType: i32,
    pub pNext: *const c_void,
    pub flags: u32,
    pub bindingCount: u32,
    pub pBindings: *const VkDescriptorSetLayoutBinding,
}
impl VkDescriptorSetLayoutCreateInfo {
    pub fn new() -> Self {
        // All other fields are C scalars, handles or pointers; callers populate required fields.
        Self {
            sType: 32,
            ..unsafe { core::mem::zeroed() }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct VkDescriptorPoolSize {
    pub r#type: i32,
    pub descriptorCount: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct VkDescriptorPoolCreateInfo {
    pub sType: i32,
    pub pNext: *const c_void,
    pub flags: u32,
    pub maxSets: u32,
    pub poolSizeCount: u32,
    pub pPoolSizes: *const VkDescriptorPoolSize,
}
impl VkDescriptorPoolCreateInfo {
    pub fn new() -> Self {
        // All other fields are C scalars, handles or pointers; callers populate required fields.
        Self {
            sType: 33,
            ..unsafe { core::mem::zeroed() }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct VkDescriptorSetAllocateInfo {
    pub sType: i32,
    pub pNext: *const c_void,
    pub descriptorPool: u64,
    pub descriptorSetCount: u32,
    pub pSetLayouts: *const u64,
}
impl VkDescriptorSetAllocateInfo {
    pub fn new() -> Self {
        // All other fields are C scalars, handles or pointers; callers populate required fields.
        Self {
            sType: 34,
            ..unsafe { core::mem::zeroed() }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct VkDescriptorImageInfo {
    pub sampler: u64,
    pub imageView: u64,
    pub imageLayout: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct VkDescriptorBufferInfo {
    pub buffer: u64,
    pub offset: u64,
    pub range: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct VkWriteDescriptorSet {
    pub sType: i32,
    pub pNext: *const c_void,
    pub dstSet: u64,
    pub dstBinding: u32,
    pub dstArrayElement: u32,
    pub descriptorCount: u32,
    pub descriptorType: i32,
    pub pImageInfo: *const VkDescriptorImageInfo,
    pub pBufferInfo: *const VkDescriptorBufferInfo,
    pub pTexelBufferView: *const u64,
}
impl VkWriteDescriptorSet {
    pub fn new() -> Self {
        // All other fields are C scalars, handles or pointers; callers populate required fields.
        Self {
            sType: 35,
            ..unsafe { core::mem::zeroed() }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct VkComputePipelineCreateInfo {
    pub sType: i32,
    pub pNext: *const c_void,
    pub flags: u32,
    pub stage: VkPipelineShaderStageCreateInfo,
    pub layout: u64,
    pub basePipelineHandle: u64,
    pub basePipelineIndex: i32,
}
impl VkComputePipelineCreateInfo {
    pub fn new() -> Self {
        // All other fields are C scalars, handles or pointers; callers populate required fields.
        Self {
            sType: 29,
            ..unsafe { core::mem::zeroed() }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct VkPipelineDynamicStateCreateInfo {
    pub sType: i32,
    pub pNext: *const c_void,
    pub flags: u32,
    pub dynamicStateCount: u32,
    pub pDynamicStates: *const i32,
}
impl VkPipelineDynamicStateCreateInfo {
    pub fn new() -> Self {
        // All other fields are C scalars, handles or pointers; callers populate required fields.
        Self {
            sType: 27,
            ..unsafe { core::mem::zeroed() }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct VkImageBlit {
    pub srcSubresource: VkImageSubresourceLayers,
    pub srcOffsets: [VkOffset3D; 2],
    pub dstSubresource: VkImageSubresourceLayers,
    pub dstOffsets: [VkOffset3D; 2],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct VkLayerProperties {
    pub layerName: [c_char; 256],
    pub specVersion: u32,
    pub implementationVersion: u32,
    pub description: [c_char; 256],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct VkExtensionProperties {
    pub extensionName: [c_char; 256],
    pub specVersion: u32,
}

// VK_EXT_debug_utils (instance extension; entry points come from vkGetInstanceProcAddr).
pub(in crate::vulkan) type PFN_vkVoidFunction = unsafe extern "system" fn();
pub(in crate::vulkan) type PFN_vkDebugUtilsMessengerCallbackEXT = unsafe extern "system" fn(
    messageSeverity: u32,
    messageTypes: u32,
    pCallbackData: *const VkDebugUtilsMessengerCallbackDataEXT,
    pUserData: *mut c_void,
) -> u32;
pub(in crate::vulkan) type PFN_vkCreateDebugUtilsMessengerEXT = unsafe extern "system" fn(
    instance: *mut c_void,
    pCreateInfo: *const VkDebugUtilsMessengerCreateInfoEXT,
    pAllocator: *const c_void,
    pMessenger: *mut u64,
) -> i32;
pub(in crate::vulkan) type PFN_vkDestroyDebugUtilsMessengerEXT =
    unsafe extern "system" fn(instance: *mut c_void, messenger: u64, pAllocator: *const c_void);

#[repr(C)]
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct VkDebugUtilsMessengerCreateInfoEXT {
    pub sType: i32,
    pub pNext: *const c_void,
    pub flags: u32,
    pub messageSeverity: u32,
    pub messageType: u32,
    pub pfnUserCallback: Option<PFN_vkDebugUtilsMessengerCallbackEXT>,
    pub pUserData: *mut c_void,
}
impl VkDebugUtilsMessengerCreateInfoEXT {
    pub fn new() -> Self {
        // All other fields are C scalars, handles or pointers; callers populate required fields.
        Self {
            sType: 1000128004,
            ..unsafe { core::mem::zeroed() }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct VkDebugUtilsMessengerCallbackDataEXT {
    pub sType: i32,
    pub pNext: *const c_void,
    pub flags: u32,
    pub pMessageIdName: *const c_char,
    pub messageIdNumber: i32,
    pub pMessage: *const c_char,
    pub queueLabelCount: u32,
    pub pQueueLabels: *const c_void,
    pub cmdBufLabelCount: u32,
    pub pCmdBufLabels: *const c_void,
    pub objectCount: u32,
    pub pObjects: *const c_void,
}

// VK_EXT_validation_features (instance extension provided by the validation layer).
#[repr(C)]
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct VkValidationFeaturesEXT {
    pub sType: i32,
    pub pNext: *const c_void,
    pub enabledValidationFeatureCount: u32,
    pub pEnabledValidationFeatures: *const i32,
    pub disabledValidationFeatureCount: u32,
    pub pDisabledValidationFeatures: *const i32,
}
impl VkValidationFeaturesEXT {
    pub fn new() -> Self {
        // All other fields are C scalars, handles or pointers; callers populate required fields.
        Self {
            sType: 1000247000,
            ..unsafe { core::mem::zeroed() }
        }
    }
}
