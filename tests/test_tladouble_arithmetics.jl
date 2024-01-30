using ADOLC.TladoubleModule
using Test
using IrrationalConstants


#### test with passie adouble type

a = Tladouble(3.0)
@test (a + 3.0).val == 6.0
@test typeof(a + 3.0) == Tladouble
@test typeof((a + 3.0).val) == Float64
@test typeof((a + invsqrt2).val) == Float64
@test (a + invsqrt2 - invsqrt2).val == 3.0

#### test with with active adouble type

a = Tladouble(3.0, true)
@test (a + 3.0).val == 6.0
@test typeof(a + 3.0) == Tladouble
@test typeof((a + 3.0).val) == TladoubleModule.TladoubleCxxAllocated
@test typeof((a + invsqrt2).val) == TladoubleModule.TladoubleCxxAllocated
@test (a + invsqrt2 - invsqrt2).val == 3.0


#### test with with active and passive adouble type

b = Tladouble(-3.6)
@test isapprox(getValue((a + b).val), -0.6)
@test typeof((a + b).val) == TladoubleModule.TladoubleCxxAllocated
@test typeof((a + b)) == Tladouble
