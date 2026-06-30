// RUN: aster-opt %s --pass-pipeline="builtin.module(amdgcn-wrap-in-module{target=gfx942})" | FileCheck %s --check-prefix=CHECK-DEFAULT
// RUN: aster-opt %s --pass-pipeline="builtin.module(amdgcn-wrap-in-module{target=gfx1250 name=my_mod})" | FileCheck %s --check-prefix=CHECK-NAMED

// CHECK-DEFAULT-LABEL: module
// CHECK-DEFAULT:         amdgcn.module @module target = <gfx942> {
// CHECK-DEFAULT-NEXT:      func.func @foo() {
// CHECK-DEFAULT-NEXT:        return
// CHECK-DEFAULT-NEXT:      }
// CHECK-DEFAULT-NEXT:      func.func @bar() {
// CHECK-DEFAULT-NEXT:        return
// CHECK-DEFAULT-NEXT:      }
// CHECK-DEFAULT-NEXT:    }

// CHECK-NAMED-LABEL: module
// CHECK-NAMED:         amdgcn.module @my_mod target = <gfx1250> {
// CHECK-NAMED-NEXT:      func.func @foo() {
// CHECK-NAMED-NEXT:        return
// CHECK-NAMED-NEXT:      }
// CHECK-NAMED-NEXT:      func.func @bar() {
// CHECK-NAMED-NEXT:        return
// CHECK-NAMED-NEXT:      }
// CHECK-NAMED-NEXT:    }

module {
  func.func @foo() {
    return
  }
  func.func @bar() {
    return
  }
}
