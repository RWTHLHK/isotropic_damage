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

function new_state(epsilon, p_in, epsilonp_in) #p_in effective_plastic_strain epsilonp_in plastin strain tensor
  sigma_trial = sigmae(epsilon)
  sigma_tdev = dev(sigma_trial) # compute deviatoric trial stress
  mises_sigmat = mises_stress(sigma_tdev) # compute trial mises stress
  # epsilonp_in = assemble_sym_matrix(epp11,epp22,epp33,epp12,epp13,epp23)
  if check_yield(mises_sigmat, p_in) < sigma0*ytol
    yield = false
    delp = 0
    p_out = p_in
    epsilonp_out = epsilonp_in
  else
    yield = true
    delp = solve_p(mises_sigmat, p_in) #newton raphson method
    flow_n = sigma_tdev / mises_sigmat # flow direction
    p_out = p_in + delp
    epsilonp_out = epsilonp_in + 3.0/2.0 * delp * flow_n
  end
  yield, delp, p_out, epsilonp_out
end

function sigma(epsilon_in, p_in,  epsilonp_in)
  _,delp,p_out,epsilonp_out = new_state(epsion_in, p_in,  epsilonp_in)
  sigmae(epsilon_in - epsilonp_out)
end

function dsigma(epsilon_inmdeps, state)
  yield, delp, p_out, epsilonp = state
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

const b_max = VectorValue(0.0,0.0,-(9.81*2.5e3))

# ## L2 projection
# form Gauss points to a Lagrangian piece-wise discontinuous space
function project(q,model,dΩ,order)
  reffe = ReferenceFE(lagrangian,Float64,order)
  V = FESpace(model,reffe,conformity=:L2)
  a(u,v) = ∫( u*v )*dΩ
  l(v) = ∫( v*q )*dΩ
  op = AffineFEOperator(a,l,V,V)
  qh = solve(op)
  qh
end

function main(;n,nsteps)

  r = 12
  domain = (0,r,0,1,0,1)
  partition = (r*n,n,n)
  model = CartesianDiscreteModel(domain,partition)

  labeling = get_face_labeling(model)
  add_tag_from_tags!(labeling,"supportA",[1,3,5,7,13,15,17,19,25])
  add_tag_from_tags!(labeling,"supportB",[2,4,6,8,14,16,18,20,26])
  add_tag_from_tags!(labeling,"supports",["supportA","supportB"])

  order = 1

  reffe = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
  V = TestFESpace(model,reffe,labels=labeling,dirichlet_tags=["supports"])
  U = TrialFESpace(V)

  degree = 2*order
  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)

  p = CellState(0.0,dΩ)
  nulls = Gridap.TensorValues.SymTensorValue{3, Float64, 6}(0,0,0,0,0,0)
  epsilonp = CellState(nulls, dΩ)
  nls = NLSolver(show_trace=true, method=:newton)
  solver = FESolver(nls)

  function step(uh_in,factor,cache)
    b = factor*b_max
    res(u,v) = ∫(  ε(v) ⊙ (σ∘(ε(u),p, epsilonp))  - v⋅b )*dΩ
    jac(u,du,v) = ∫(  ε(v) ⊙ (dσ∘(ε(u), ε(du),new_state∘(ε(u),p, epsilonp)))  )*dΩ
    op = FEOperator(res,jac,U,V)
    uh_out, cache = solve!(uh_in,solver,op,cache)
    update_state!(new_state,r,d,ε(uh_out))
    uh_out, cache
  end

  factors = collect(1:nsteps)*(1/nsteps)
  uh = zero(V)
  cache = nothing
  for (istep,factor) in enumerate(factors)

    println("\n+++ Solving for load factor $factor in step $istep of $nsteps +++\n")

    uh,cache = step(uh,factor,cache)
    ph = project(d,model,dΩ,order)

    writevtk(
      Ω,"results_$(lpad(istep,3,'0'))",
      cellfields=["uh"=>uh,"epsi"=>ε(uh),"p"=>ph])

  end

end

main(n=6,nsteps=1)
