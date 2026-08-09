`timescale 1ns/1ps

module pc #(
    parameter int PC_W = 7
)(
    input  logic clk, rst,
    input  logic clr,          // 1 cycle on each start → pc = 0
    input  logic inc,          // during FETCH1 → pc++
    output logic [PC_W-1:0] pc_addr
);

logic [PC_W-1:0] pc_reg;

always_ff @(posedge clk or posedge rst) begin
    if (rst || clr) pc_reg <= '0;
    else if (inc)   pc_reg <= pc_reg + 1'b1;
end

assign pc_addr = pc_reg;

endmodule