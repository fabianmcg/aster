// RUN: aster-opt %s --pass-pipeline="builtin.module(amdgcn-wrap-in-module{target=badtarget})" --verify-diagnostics

// expected-error @below {{unknown AMDGCN target 'badtarget'}}
module {
}
