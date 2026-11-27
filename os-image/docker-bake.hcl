variable "INKLING2_ENGINE_CONTEXT" {
  default = ""
}

target "private-cuda-runtime" {
  context    = "."
  dockerfile = "os-image/Dockerfile.cuda-builder"
  target     = "private-cuda-runtime"
  contexts = {
    engine = INKLING2_ENGINE_CONTEXT
  }
}

target "private-disk-tools" {
  context    = "."
  dockerfile = "os-image/Dockerfile.iso-builder"
  target     = "private-disk-tools"
  contexts = {
    private-cuda = "target:private-cuda-runtime"
  }
  tags = ["inkling2-private-disk:local"]
}
