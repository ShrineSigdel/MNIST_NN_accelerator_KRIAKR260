`timescale  1ns/1ps


module bram # (
    parameter int DATA_W = 32,
    parameter int ADDR_W = 8 //  (2^8 = 256 locations)
    parameter string INIT_FILE = ""
)(
    input logic clk,
    input logic we,
    input logic [ADDR_W-1:0] addr,
    input logic [DATA_W-1:0] data_in,
    output logic [DATA_W-1:0] data_out,
);

// Create a memory array to store the data 
(* ram_style = "block" *) logic [DATA_W-1:0] mem [(1<<ADDR_W)-1:0]; // 2^ADDR_W locations

initial begin
    if (INIT_FILE != "") begin
        $readmemh(INIT_FILE, mem);
    end
end

always_ff @(posedge clk) begin
    if (we) begin
        mem[addr] <= data_in; // Write operation
    end
    data_out <= mem[addr]; // Read operation
end

endmodule;