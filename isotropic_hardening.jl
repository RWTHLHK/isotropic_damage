using Gridap
using LinearAlgebra
using Test
using NLsolve

const E = 3.0e10 # Pa
const v = 0.3
const lamda = (E*v)/(1+v)*(1-2*v)
const mu = E/(2*(1+v))
const sigma0 = 4.0e8 #Pa
const ytol = 1e-9 # yield tolerence
const h = 3.0e5 #Pa
sigmae(epsilon) = lamda * tr(epsilon) * one(epsilon) + 2.0*mu*epsilon

function dev(m)
  m - (tr(m)/3.0)*one(m)
end

function mises_stress(sigma_dev)
  sqrt(3.0/2.0 * sum(sigma_dev.*sigma_dev))
end

function hard_func(p) #linear isotropic hardening
  sigma0 + h*p
end

function hard_derivative(p)
  h
end
function check_yield(mises, p)
  mises - hard_func(p)
end

function new_state(epsilon, deps, p_in)
  sigma_trial = simgae(epsilon + deps)
  sigma_tdev = dev(sigma_trial) # compute deviatoric trial stress
  mises_sigmat = mises_stress(sigma_tdev) # compute trial mises stress
  flow_n = sigma_tdev / mises_sigmat # flow direction
  if check_yield(mises_sigmat, p) < sigma0*ytol
    yield = false
    delp = 0
    p_out = p_in
  else
    yield = true
    delp = solve_p(mises_sigmat, p_in)
    p_out = p_in + delp
  end
  yield, delp, p_out, flow_n
end

function sigma(epsion_in, deps, p_in)
  _,delp,p_out,flow_n = new_state(epsion_in, deps, p_in)
  depsp = 3.0/2.0 * delp * flow_n # plastic strain Incremental
  depse = deps - depsp #elastic strain increment
  sigmae(depse)
end

function dsigma(epsilon, deps, state)
  yield, delp, p_out, flow_n = state
  if yield
    depsp = 3.0/2.0 * delp * flow_n # plastic strain Incremental
    depse = deps - depsp #elastic strain increment
    sigmae(depse)
  else
    sigmae(deps)
  end
end

function solve_p(mises, p_in)
  delp = 0
  kewton = 1000
  epoch = 0
  yield_residual(delp) = mises - hard_func(p_in+delp) - 3*mu*delp
  yield_jac(delp) = -3*mu - h
  res = yield_residual(delp)
  jac = yield_jac(delp)
  while res > sigma0*ytol && epoch < kewton
    delp -= res/jac
    res = yield_residual(delp)
    jac = yield_jac(delp)
    epoch += 1
  end
  delp
end


function test_yield()
    # Define some test inputs
    epsilon = [0.001 0.0 0.0; 0.0 -0.0005 0.0; 0.0 0.0 -0.0005] # Small strain tensor
    deps = [0.0001 0.0 0.0; 0.0 -0.00005 0.0; 0.0 0.0 -0.00005] # Incremental strain tensor
    p = 0.0

    # Expected behavior: compute yield using the given inputs
    # Compute trial stress tensor
    sigma_trial = sigmae(epsilon + deps)

    # Compute deviatoric trial stress
    sigma_tdev = dev(sigma_trial)

    # Compute trial von Mises stress
    mises_sigmat = mises_stress(sigma_tdev)

    # Compute the expected yield value
    expected_yield = mises_sigmat - sigma0
    println("yield value is $expected_yield")

    # Test the yield function
    @test check_yield(mises_sigmat, p) ≈ expected_yield atol=1e-6
end

function test_nlsolve()
  # delp = solve_p(7e8, 0.0)
  # println("delta p is $delp")
  delp = 0
  p_in = 0.0
  mises = 1.0e9
  dp = solve_p(mises, p_in)
  @test mises - hard_func(p_in + dp) - 3*mu*dp < sigma0*ytol
end
# Run the test
test_yield()
test_nlsolve()


