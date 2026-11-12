// Minimal WebGPU browser global declarations so `web/sdk` compiles offline under strict TypeScript
// without requiring `@webgpu/types` in `node_modules`.

export interface WebGpuNavigator {
  readonly gpu?: {
    requestAdapter(options?: { powerPreference?: string }): Promise<WebGpuAdapter | null>;
    getPreferredCanvasFormat(): string;
  };
}

export interface WebGpuAdapter {
  readonly features: ReadonlySet<string>;
  readonly limits: {
    readonly maxBufferSize: number;
    readonly maxStorageBufferBindingSize: number;
    readonly maxTextureDimension2D: number;
    readonly maxTextureArrayLayers: number;
    readonly maxBindGroups: number;
    readonly minUniformBufferOffsetAlignment: number;
    readonly minStorageBufferOffsetAlignment: number;
    readonly maxUniformBufferBindingSize: number;
    readonly maxColorAttachments: number;
    readonly maxComputeWorkgroupsPerDimension: number;
    readonly maxVertexBufferArrayStride: number;
    readonly maxVertexAttributes: number;
  };
  readonly info?: {
    readonly vendor?: string;
    readonly architecture?: string;
    readonly device?: string;
    readonly description?: string;
  };
  requestDevice(descriptor?: {
    requiredFeatures?: string[];
    requiredLimits?: Record<string, number>;
  }): Promise<WebGpuDevice>;
}

export interface WebGpuTexture {
  readonly width: number;
  readonly height: number;
  readonly depthOrArrayLayers: number;
  readonly mipLevelCount: number;
  createView(descriptor?: Record<string, unknown>): unknown;
  destroy(): void;
}

export interface WebGpuBuffer {
  destroy(): void;
  mapAsync(mode: number, offset?: number, size?: number): Promise<void>;
  getMappedRange(offset?: number, size?: number): ArrayBuffer;
  unmap(): void;
}

export interface WebGpuRenderPassEncoder {
  setPipeline(pipeline: unknown): void;
  setBindGroup(
    index: number,
    bindGroup: unknown,
    dynamicOffsetsData?: Uint32Array,
    dynamicOffsetsDataStart?: number,
    dynamicOffsetsDataLength?: number,
  ): void;
  setVertexBuffer(slot: number, buffer: unknown, offset?: number): void;
  setIndexBuffer(buffer: unknown, format: 'uint32' | 'uint16', offset?: number): void;
  setViewport(x: number, y: number, width: number, height: number, minDepth: number, maxDepth: number): void;
  setScissorRect(x: number, y: number, width: number, height: number): void;
  draw(vertexCount: number, instanceCount?: number, firstVertex?: number, firstInstance?: number): void;
  drawIndexed(
    indexCount: number,
    instanceCount?: number,
    firstIndex?: number,
    baseVertex?: number,
    firstInstance?: number,
  ): void;
  executeBundles(bundles: unknown[]): void;
  end(): void;
}

export interface WebGpuComputePassEncoder {
  setPipeline(pipeline: unknown): void;
  setBindGroup(
    index: number,
    bindGroup: unknown,
    dynamicOffsetsData?: Uint32Array,
    dynamicOffsetsDataStart?: number,
    dynamicOffsetsDataLength?: number,
  ): void;
  dispatchWorkgroups(x: number, y?: number, z?: number): void;
  end(): void;
}

export interface WebGpuRenderBundleEncoder {
  setPipeline(pipeline: unknown): void;
  setBindGroup(
    index: number,
    bindGroup: unknown,
    dynamicOffsetsData?: Uint32Array,
    dynamicOffsetsDataStart?: number,
    dynamicOffsetsDataLength?: number,
  ): void;
  setVertexBuffer(slot: number, buffer: unknown, offset?: number): void;
  setIndexBuffer(buffer: unknown, format: 'uint32' | 'uint16', offset?: number): void;
  draw(vertexCount: number, instanceCount?: number, firstVertex?: number, firstInstance?: number): void;
  drawIndexed(
    indexCount: number,
    instanceCount?: number,
    firstIndex?: number,
    baseVertex?: number,
    firstInstance?: number,
  ): void;
  finish(): unknown;
}

export interface WebGpuCommandEncoder {
  beginRenderPass(descriptor: Record<string, unknown>): WebGpuRenderPassEncoder;
  beginComputePass(): WebGpuComputePassEncoder;
  copyBufferToBuffer(
    source: unknown,
    sourceOffset: number,
    destination: unknown,
    destinationOffset: number,
    size: number,
  ): void;
  copyTextureToBuffer(
    source: Record<string, unknown>,
    destination: Record<string, unknown>,
    copySize: [number, number, number],
  ): void;
  copyTextureToTexture(
    source: Record<string, unknown>,
    destination: Record<string, unknown>,
    copySize: [number, number, number],
  ): void;
  resolveQuerySet(
    querySet: unknown,
    firstQuery: number,
    queryCount: number,
    destination: unknown,
    destinationOffset: number,
  ): void;
  finish(): unknown;
}

export interface WebGpuShaderModule {
  readonly label?: string;
  getCompilationInfo(): Promise<{
    readonly messages: readonly {
      readonly type: string;
      readonly message: string;
      readonly lineNum?: number;
      readonly linePos?: number;
    }[];
  }>;
}

export interface WebGpuDevice {
  readonly features: ReadonlySet<string>;
  readonly limits: WebGpuAdapter['limits'];
  readonly lost: Promise<{ readonly reason: string; readonly message: string }>;
  readonly queue: {
    writeBuffer(
      buffer: unknown,
      bufferOffset: number,
      data: ArrayBufferLike,
      dataOffset?: number,
      size?: number,
    ): void;
    writeTexture(
      destination: Record<string, unknown>,
      data: ArrayBufferLike,
      dataLayout: Record<string, unknown>,
      size: [number, number, number],
    ): void;
    copyExternalImageToTexture(
      source: Record<string, unknown>,
      destination: Record<string, unknown>,
      copySize: [number, number],
    ): void;
    submit(commandBuffers: unknown[]): void;
    onSubmittedWorkDone(): Promise<void>;
  };
  addEventListener(type: 'uncapturederror', listener: (event: { readonly error: { readonly message: string } }) => void): void;
  createBuffer(descriptor: { size: number; usage: number }): WebGpuBuffer;
  createTexture(descriptor: Record<string, unknown>): WebGpuTexture;
  createSampler(descriptor: Record<string, unknown>): unknown;
  createShaderModule(descriptor: { code: string; label?: string }): WebGpuShaderModule;
  createBindGroupLayout(descriptor: { entries: Record<string, unknown>[] }): unknown;
  createPipelineLayout(descriptor: { bindGroupLayouts: unknown[] }): unknown;
  createRenderPipeline(descriptor: Record<string, unknown>): { readonly label?: string };
  createComputePipeline(descriptor: Record<string, unknown>): unknown;
  createBindGroup(descriptor: { layout: unknown; entries: Record<string, unknown>[] }): unknown;
  createCommandEncoder(): WebGpuCommandEncoder;
  createRenderBundleEncoder?(descriptor: Record<string, unknown>): WebGpuRenderBundleEncoder;
  createQuerySet(descriptor: { type: 'timestamp'; count: number }): unknown;
}

export interface WebGpuCanvasContext {
  configure(configuration: Record<string, unknown>): void;
  getCurrentTexture(): WebGpuTexture;
}

declare global {
  var GPUTextureUsage: {
    readonly COPY_SRC: number;
    readonly COPY_DST: number;
    readonly TEXTURE_BINDING: number;
    readonly STORAGE_BINDING: number;
    readonly RENDER_ATTACHMENT: number;
  };
  var GPUBufferUsage: {
    readonly MAP_READ: number;
    readonly MAP_WRITE: number;
    readonly COPY_SRC: number;
    readonly COPY_DST: number;
    readonly INDEX: number;
    readonly VERTEX: number;
    readonly UNIFORM: number;
    readonly STORAGE: number;
    readonly INDIRECT: number;
    readonly QUERY_RESOLVE: number;
  };
  var GPUShaderStage: {
    readonly VERTEX: number;
    readonly FRAGMENT: number;
    readonly COMPUTE: number;
  };
  var GPUMapMode: {
    readonly READ: number;
    readonly WRITE: number;
  };
}


