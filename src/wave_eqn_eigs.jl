using LinearAlgebra
using Arpack
using Gridap
using Gridap: ∇

function H₀(x)
  H = 1.0 + 0.1*cos(x-π)
  H
end

function dH₀(x)
  dH = 1.0 - 0.1*sin(x-π)
  dH
end

function s₀(x)
  H = H₀(x)
  s = 1.0/H/H
  s
end

function ds₀(x)
  H = H₀(x)
  dH= dH(x)
  ds = -2.0*dH/H/H/H
  ds
end

function S₀(x)
  H = H₀(x)
  s = s₀(x)
  S = s*H
  S
end

function profiles(x)
  H  = 1.0 .+ 0.1.*cos.(x .- π)
  dH = 1.0 .- 0.1.*sin.(x .- π)
  s  = 1.0 ./ H ./ H
  ds = -2.0 .* dH ./ H ./ H ./ H
  S  = s .* H
  H, dH, s, ds, S
end

D = 2*π
order = 1
domain = (0,D)
partition = (128,)
model = CartesianDiscreteModel(domain, partition)
Ω = Triangulation(model)
dΩ = Measure(Ω, 4*order)

V = FESpace(model, ReferenceFE(lagrangian, Float64, order), conformity=:H1)
Q = FESpace(model, ReferenceFE(lagrangian, Float64, order-1), conformity=:L2)
U = TrialFESpace(V)
P = TrialFESpace(Q)

H(x)  = profiles(x[1])[1]
dH(x) = profiles(x[1])[2]
s(x)  = profiles(x[1])[3]
ds(x) = profiles(x[1])[4]
S(x)  = profiles(x[1])[5]

mh(p,q) = ∫(q*p)dΩ
Mu = assemble_matrix(mh, P, Q)

mu(u,v) = ∫(v⋅u)dΩ
Mq = assemble_matrix(mu, U, V)

md(u,q) = ∫(q⋅∇(u))dΩ
DIV = assemble_matrix(md, U, Q)
#b(q) = 0
#op = AffineFEOperator(md,b,U,Q)
#D=op.op.matrix

mg(p,v) = ∫(∇(v)⋅p)dΩ
GRAD = assemble_matrix(mg, P, V)

MqInv = inv(Matrix(Mq))
MuInv = inv(Matrix(Mu))

A = MuInv*GRAD*MqInv*DIV

ω,v = eigs(A; nev=122)
ωr = real(ω)
ωi = imag(ω)

#mh(a,b) = ∫(a⋅b*H)dΩ
#Muh = assemble_matrix(mh, U, V)

