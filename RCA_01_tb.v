`timescale 1ns / 1ps

module RCA_01_tb();

parameter N = 8;

//wire [N-1:0] G,P;
reg [N:1] a,b;
reg cin;
wire cout;
wire [N:1] sum;

RCA_01 RCA01(sum,cout,a,b,cin);

initial begin
    a = 8'h10;
    b = 8'h11;
    cin = 1'b0;
    #10 a = 8'h16; b = 8'h20; cin = 1'b1;
    #20 $finish;
end

endmodule
