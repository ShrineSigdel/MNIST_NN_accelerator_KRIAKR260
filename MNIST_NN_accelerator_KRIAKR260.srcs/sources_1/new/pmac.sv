`timescale  1ns/1ps

module pmac # (
    parameter int DATA_W = 8,
    parameter int ACC_W = 32
)(
    input logic clk, rst,
    input logic acc_clr,
    input logic  [DATA_W-1:0] a_in , b_in,
    output logic  [DATA_W-1:0] a_out, b_out,
    output logic  [ACC_W-1:0] acc_out
);

logic [ACC_W-1:0] acc_reg, acc_next;

// Sequential logic 
always_ff @(posedge clk or posedge rst) begin
    if (rst) begin
        acc_reg <= '0;
        a_out <= '0;
        b_out <= '0;
    end else begin
        acc_reg <= acc_next;
        a_out <= a_in;
        b_out <= b_in;
    end
end

assign acc_out = acc_reg;
assign acc_next = acc_clr ? '0 : (acc_reg + a_in * b_in);

endmodule