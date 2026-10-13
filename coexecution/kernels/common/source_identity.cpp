// Bind the reviewed source snapshot to each owned extension registration.
#include <string>
#include <torch/library.h>

#ifndef PIR2_KERNEL_NAMESPACE
#error "PIR2_KERNEL_NAMESPACE is required"
#endif
#ifndef PIR2_BUILD_SOURCE_ID
#error "PIR2_BUILD_SOURCE_ID is required"
#endif

TORCH_LIBRARY_FRAGMENT(PIR2_KERNEL_NAMESPACE, m) {
  m.def("source_identity() -> str",
        []() -> std::string { return PIR2_BUILD_SOURCE_ID; });
}
