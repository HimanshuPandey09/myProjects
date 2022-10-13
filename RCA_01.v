`timescale 1ns / 1ps

module RCA_01(output sum,
              output cout,
              input a,
              input b,
              input cin);
              
              
parameter N = 8;
              
wire [N:1] a, b;
wire [N:0] g, p;
wire [N-1:0] G, P;
wire [N:1] sum; 
              
assign g[0] = cin,
       p[0] = 1'b0;
              
genvar i;
              
generate for(i = 1; i < N + 1; i = i + 1) begin: bitPG
    assign g[i] = a[i] & b[i],
           p[i] = a[i] ^ b[i];
                      
end
endgenerate

assign G[0] = g[0],
       P[0] = 1'b0;
genvar j;
generate for (j = 1; j < N; j = j+1) begin: groupPG
    assign G[j] = g[j] | (p[j] & G[j-1]);
end
endgenerate


genvar k;
generate for (k = 1; k < N+1; k = k+1) begin: sumLogic

    assign sum[k] = p[k] ^ G[k-1];
    
    if (k == N)
        assign cout = (g[N] | (p[N] & G[N-1]));
    
end
endgenerate
endmodule
