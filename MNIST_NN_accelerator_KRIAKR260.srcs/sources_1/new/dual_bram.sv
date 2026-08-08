`timescale  1ns/1ps

/*
    Dual port and dual block RAM inference for FPGA.
*/


module bram # (
    parameter int DATA_W = 32,
    parameter int ADDR_W = 8 //  (2^8 = 256 locations)
)(
    // Port A inference
    input logic clk_a,
    input logic we_a,
    input logic [ADDR_W-1:0] addr_a,
    input logic [DATA_W-1:0] data_in_a,
    output logic [DATA_W-1:0] data_out_a,

    // Port B inference
    input logic clk_b,
    input logic we_b,
    input logic [ADDR_W-1:0] addr_b,
    input logic [DATA_W-1:0] data_in_b,
    output logic [DATA_W-1:0] data_out_b
);

// Create a memory array to store the data (32*2 = 64 kbits)
(* ram_style = "block" *) logic [DATA_W-1:0] mem_a [(1<<ADDR_W)-1:0]; // 2^ADDR_W locations
(* ram_style = "block" *) logic [DATA_W-1:0] mem_b [(1<<ADDR_W)-1:0]; // 2^ADDR_W locations


// Todo: Initialize the memory with a.mem and b.mem

// Port A operations
always_ff @(posedge clk_a) begin
    if (we_a) begin
        mem_a[addr_a] <= data_in_a; // Write operation
    end
    data_out_a <= mem_a[addr_a]; // Read operation
end

// Port B operations
always_ff @(posedge clk_b) begin
    if (we_b) begin
        mem_b[addr_b] <= data_in_b; // Write operation
    end
    data_out_b <= mem_b[addr_b]; // Read operation
end


endmodule;