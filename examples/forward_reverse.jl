include("../src/ADOLC.jl")
using .ADOLC.array_types
using .ADOLC.AdoubleModule


n = 4
X = myalloc2(1,n+4);
Y = myalloc2(1,n+4);
X[1, 1] = 1.0;                
X[1, 2] = 1.0;  

for i=1:n+2
  X[1, i+2] = 0.0
end  
Z = myalloc2(1,n+2)


# declare active variables 
x = AdoubleCxx()
y = AdoubleCxx()   
# beginning of active section

# tag = 1 and keep = 0
trace_on(1); 

# only one independent var
x << X[1, 1];             


#actual function call
y = x^n;

#only one dependent adouble
y >> Y[1, 1]
trace_off(0); 

u = alloc_vec_double(1)
u[1] = 1.0                       
for i = 1:n+2            
  ad_forward(1,1,1,i,i+1,X,Y); 
  if i == 1   
    println(Y[1, i], " - ", getValue(y), " = ", Y[1, i]- getValue(y)," (should be 0)")

  else
    Z[1, i] = Z[1, i-1]/(i - 1)
    println(Y[1, i], " - ", Z[1, i], " = ", Y[1, i]-Z[1, i], " (should be 0)")
    end
  ad_reverse(1,1,1,i,u,Z)
end

