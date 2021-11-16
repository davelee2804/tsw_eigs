using LinearAlgebra
using Plots
using Arpack
using Gridap
using Gridap: ∇

function profiles(x)
  α  = 0.5
  k  = 1
  H  = 1.0 .+ α.*cos.(k.*x .- π)
  dH = 1.0 .- k.*α.*sin.(k.*x .- π)
  e  = 1.0 ./ H ./ H
  de = -2.0 .* dH ./ H ./ H ./ H
  E  = e .* H
  f  = 0.0 .* sin.(x .- π)
  H, dH, e, de, E, f
end

D = 2*π
order = 1
domain = (0,D)
partition = (64,)

model = CartesianDiscreteModel(domain, partition; isperiodic=(true,))
Ω = Triangulation(model)
dΩ = Measure(Ω, 2*(order+1))
Γ = SkeletonTriangulation(model)
dΓ = Measure(Γ, 0)
n_Γ = get_normal_vector(Γ)

S = FESpace(model, ReferenceFE(lagrangian, Float64, order+1), conformity=:H1)
V = FESpace(model, ReferenceFE(raviart_thomas, Float64, order), conformity=:HDiv)
Q = FESpace(model, ReferenceFE(lagrangian, Float64, order), conformity=:L2)
R = TrialFESpace(S)
U = TrialFESpace(V)
P = TrialFESpace(Q)

h(x)  = profiles(x[1])[1]
dh(x) = profiles(x[1])[2]
e(x)  = profiles(x[1])[3]
de(x) = profiles(x[1])[4]
E(x)  = profiles(x[1])[5]
f(x)  = profiles(x[1])[6]

h2 = interpolate_everywhere(h,Q)
e0 = interpolate_everywhere(e,S)
e2 = interpolate_everywhere(e,Q)
E2 = interpolate_everywhere(E,Q)

mp(p,q) = ∫(q*p)dΩ
Mp = assemble_matrix(mp, P, Q)
MpInv = inv(Matrix(Mp))

mu(u,v) = ∫(v⋅u)dΩ
Mu = assemble_matrix(mu, U, V)
MuInv = inv(Matrix(Mu))

ms(r,s) = ∫(s*r)dΩ
Ms = assemble_matrix(ms, R, S)
MsInv = inv(Matrix(Ms))

md(u,q) = ∫(q*(∇⋅u))dΩ
D = assemble_matrix(md, U, Q)
G = transpose(D)

muc(u,v) = ∫(f*u⋅v)dΩ
Muc = assemble_matrix(muc, U, V)

# wave eqn
A = MuInv*G*MpInv*D + MuInv*Muc*MuInv*Muc
ω1,v1 = eigs(A; nev=(order+1)*partition[1]-2)
ω1r = real(ω1)
ω1i = imag(ω1)

# Flux form, E∈ L²(Ω)
msp(p,s) = ∫(s*p)dΩ
Msp = assemble_matrix(msp, P, S)
Mps = transpose(Msp)

muh(u,v) = ∫(v⋅u*h2)dΩ
Muh = assemble_matrix(muh, U, V)
mue2(u,v) = ∫(v⋅u*e2)dΩ
Mue2 = assemble_matrix(mue2, U, V)

Ge2 = Mue2*MuInv*G
Dh = D*MuInv*Muh
De2 = D*MuInv*Mue2*MuInv*Muh

FFE2 = 0.5*MuInv*Ge2*MpInv*Dh + 0.5*MuInv*G*MpInv*De2
ω2,v2 = eigs(FFE2; nev=(order+1)*partition[1]-2)
ω2r = real(ω2)
ω2i = imag(ω2)

# Flux form E∈ H¹(Ω)
mue0(u,v) = ∫(v⋅u*e0)dΩ
Mue0 = assemble_matrix(mue0, U, V)
mgs(v,r) = ∫((∇⋅v)⋅r)dΩ
Gs = assemble_matrix(mgs, R, V)
Ds = transpose(Gs)

Gse0 = Mue0*MuInv*Gs
De0 = D*MuInv*Mue0*MuInv*Muh

FFE0 = 0.5*MuInv*Gse0*MsInv*Msp*MpInv*Dh + 0.5*MuInv*G*MpInv*Mps*MsInv*Ds*MuInv*Muh
ω3,v3 = eigs(FFE0; nev=(order+1)*partition[1]-2)
ω3r = real(ω3)
ω3i = imag(ω3)

# Material form, e∈ H¹(Ω)
#MAT(u,v) = ∫(h2*v⋅u)dΩ
#rhs(v) = ∫(v⋅∇(e0))dΩ
#op = AffineFEOperator(MAT, rhs, U, V)
#deu = solve(op)

#musde(r,v) = ∫(v⋅deu*r)dΩ
#Musde = assemble_matrix(musde, R, V)
#MusdeT = transpose(Musde)
#gh(p,v) = ∫((∇⋅v)*h2*p)dΩ
#Gh = assemble_matrix(gh, P, V)

#MFe0 = 0.5*MuInv*Musde*MsInv*Msp*MpInv*Dh + MuInv*Gh*MpInv*Mps*MsInv*MusdeT*MuInv*Muh

ge0(p,v) = ∫((∇⋅v)*e0*p)dΩ
Ge0 = assemble_matrix(ge0, P, V)

#b1(p,v) = ∫(mean(e0)*jump(p*v⋅n_Γ))dΓ
#B1 = assemble_matrix(b1, P, V)
#Ge0 = Ge0 - B1

b(s) = ∫(s*h2*h2)dΩ
shsq = assemble_vector(b, S)
hsq = MsInv*sh2
gsh(r,v) = ∫((∇⋅(v*h2))*r)dΩ
Gsh = assemble_matrix(gsh, R, V)

as(u,s) = ∫(e0⋅(∇⋅(u*s)))dΩ
As = assemble_matrix(as, U, S)
#b2(u,s) = ∫(mean(e0*s)*jump(u⋅n_Γ))dΓ
#B2 = assemble_matrix(b2, U, S)
#As = As - B2

MFe0 = MuInv*Ge0*MpInv*Dh + 0.5*MuInv*Gsh*MsInv*As
ω4,v4 = eigs(MFe0; nev=(order+1)*partition[1]-2)
ω4r = real(ω4)
ω4i = imag(ω4)

# material form, e∈ L²(Ω)
ge2(p,v) = ∫((∇⋅v)*e2*p)dΩ
Ge2 = assemble_matrix(ge2, P, V)
b1(p,v) = ∫(mean(e2)*jump(p*v⋅n_Γ))dΓ
B1 = assemble_matrix(b1, P, V)
Ge2 = Ge2 - B1

gh(p,v) = ∫((∇⋅v)*h2*p)dΩ
Gh = assemble_matrix(gh, P, V)
b2(p,v) = ∫(mean(h2)*jump(p*v⋅n_Γ))dΓ
B2 = assemble_matrix(b2, P, V)
Ge2 = Ge2 - B2

aq(u,q) = ∫(e2⋅(∇⋅(u*q)))dΩ
Aq = assemble_matrix(aq, U, Q)
b3(u,q) = ∫(mean(e2)*jump(q*u⋅n_Γ))dΓ
B3 = assemble_matrix(b3, U, Q)
Aq = Aq - B3

MFe2 = MuInv*Ge2*MpInv*Dh + 0.5*MuInv*Gh*MpInv*Aq
ω5,v5 = eigs(MFe2; nev=(order+1)*partition[1]-2)
ω5r = real(ω5)
ω5i = imag(ω5)

#xq = LinRange(0, (order+1)*partition[1]-3, (order+1)*partition[1]-2)
xq = LinRange(1, (order+1)*partition[1]-2, (order+1)*partition[1]-2)
xq = reverse(xq)

aω1r = lazy_map(abs, ω1r)
aω2r = lazy_map(abs, ω2r)
aω3r = lazy_map(abs, ω3r)
aω4r = lazy_map(abs, ω4r)
aω5r = lazy_map(abs, ω5r)
y1q = lazy_map(sqrt, aω1r)
y2q = lazy_map(sqrt, aω2r)
y3q = lazy_map(sqrt, aω3r)
y4q = lazy_map(sqrt, aω4r)
y5q = lazy_map(sqrt, aω5r)
z1q = lazy_map(abs, ω1i)
z2q = lazy_map(abs, ω2i)
z3q = lazy_map(abs, ω3i)
z4q = lazy_map(abs, ω4i)
z5q = lazy_map(abs, ω5i)

print(z1q,"\n")
print(z2q,"\n")
print(z3q,"\n")
print(z4q,"\n")
print(z5q,"\n")

plt1 = plot()
plot!(plt1, 0.5*xq, 0.5*xq, legend = false)
plot!(plt1, 0.5*xq, y1q, legend = true)
plot!(plt1, 0.5*xq, y2q, legend = true, seriestype=:scatter)
plot!(plt1, 0.5*xq, y3q, legend = true)
plot!(plt1, 0.5*xq, y4q, legend = true)
plot!(plt1, 0.5*xq, y5q, legend = :topleft)
display(plt1)

#mue(u,v) = ∫(v⋅u*e0)dΩ
#Mue = assemble_matrix(mue, U, V)
#muE(u,v) = ∫(v⋅u*E2)dΩ
#MuE = assemble_matrix(muE, U, V)

#B = 0.5*MuInv*Mue*MuInv*Mg*MpInv*Md*MuInv*Muh + 0.5*MuInv*Mg*MpInv*Md*MuInv*MuE + MuInv*Muc*MuInv*Muc

# flux form, E∈ H¹(Ω)
#mge(r,v) = ∫(v⋅∇(r))dΩ
#Mge = assemble_matrix(mge, R, V)
#mde(u,s) = ∫(s*(∇⋅u))dΩ
#Mde = assemble_matrix(mde, U, S)
#mgeT(r,v) = ∫((∇⋅v)*r)dΩ
#MgeT = assemble_matrix(mgeT, R, V)
#B1 = 0.5*MuInv*Mue*MuInv*Mg*MpInv*Md*MuInv*Muh + 0.5*MuInv*MgeT*MrInv*Mde*MuInv*MuE + MuInv*Muc*MuInv*Muc
#ω5,v5 = eigs(B1; nev=(order+1)*partition[1]-2)
#ω5r = real(ω5)
#ω5i = imag(ω5)

# material form, e∈ H¹(Ω) 
#mrude(u,s) = ∫(s*∇(e0)⋅u)dΩ
#Mrude = assemble_matrix(mrude, U, S)

#C = MuInv*Mue*MuInv*Mg*MpInv*Md*MuInv*Muh + 0.5*MuInv*Muh*MuInv*Mgr*MrInv*Mrude + MuInv*Muc*MuInv*Muc
#ω3,v3 = eigs(C; nev=(order+1)*partition[1]-2)
#ω3r = real(ω3)
#ω3i = imag(ω3)

# material form, e∈ L²(Ω)
#mue2(u,v) = ∫(v⋅u*e2)dΩ
#Mue2 = assemble_matrix(mue2, U, V)
#medpu(u,q) = ∫(e2*q*(∇⋅u) + e2*u⋅∇(q))dΩ
#Medpu = assemble_matrix(medpu, U, Q)
#buqe(u,q) = ∫(mean(e2)*jump(q*u⋅n_Γ))dΓ
#Buqe = assemble_matrix(buqe, U, Q)

#D = MuInv*Mue2*MuInv*Mg*MpInv*Md*MuInv*Muh - 0.5*MuInv*Muh*MuInv*Mg*MpInv*Medpu + 0.5*MuInv*Muh*MuInv*Mg*MpInv*Buqe + MuInv*Muc*MuInv*Muc
#ω4,v4 = eigs(D; nev=(order+1)*partition[1]-2)
#ω4r = real(ω4)
#ω4i = imag(ω4)


#aω2r = lazy_map(abs, ω2r)
#aω5r = lazy_map(abs, ω5r)
#aω3r = lazy_map(abs, ω3r)
#aω4r = lazy_map(abs, ω4r)
#y2q = lazy_map(sqrt, aω2r)
#y5q = lazy_map(sqrt, aω5r)
#y3q = lazy_map(sqrt, aω3r)
#y4q = lazy_map(sqrt, aω4r)

#z2q = lazy_map(abs, ω2i)
#z5q = lazy_map(abs, ω5i)
#z3q = lazy_map(abs, ω3i)
#z4q = lazy_map(abs, ω4i)
#print(z4q,"\n")
#print(y1q[12-2]-1,"\n")
#print(y2q[12-2]-1,"\n")
#print(y3q[12-2]-1,"\n")
#print(y4q[12-2]-1,"\n")

#plot_imag=false
#if(plot_imag)
#plt2 = plot()
#plot!(plt2, xq, 2*z1q, legend = true, seriestype=scatter)
#plot!(plt2, xq, 2*z2q, legend = true)
#plot!(plt2, xq, 2*z3q, legend = true)
#plot!(plt2, xq, 2*z4q, legend=:topleft)
#else
#plot!(plt1, 0.5*xq, y3q.-0*y3q[partition[1]-2], legend = true)
#plot!(plt1, 0.5*xq, y4q.-0*y4q[partition[1]-2], legend=:topleft)
#plot!(plt1, 0.5*xq, y1q-0.5*xq.-0*y1q[partition[1]-2].-1.0, legend = true, seriestype=scatter)
#plot!(plt1, 0.5*xq, y2q-0.5*xq.-0*y2q[partition[1]-2].-1.0, legend = true, seriestype=scatter)
#plot!(plt1, 0.5*xq, y5q-0.5*xq.-0*y5q[partition[1]-2].-1.0, legend = true, seriestype=scatter)
#plot!(plt1, 0.5*xq, y3q-0.5*xq.-0*y3q[partition[1]-2].-1.0, legend = true)
#plot!(plt1, 0.5*xq, y4q-0.5*xq.-0*y4q[partition[1]-2].-1.0, legend=:topleft)
#end
