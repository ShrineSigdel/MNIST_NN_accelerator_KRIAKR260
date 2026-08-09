`timescale  1ns/1ps
module skew_buffer #(
    parameter int DATA_W = 8,
    parameter int N = 4
)(
    input logic clk, rst,
    input logic clr, 
    input logic en,
    input logic [DATA_W-1:0] data_in [N-1:0],
    output logic [DATA_W-1:0] data_out [N-1:0]
);


generate
    genvar i;
        for (i = 0; i < N; i++) begin : g_row    
            
            if (i == 0) begin : g_wire 
                assign data_out [i] = data_in[i];
            end 

            else begin 
                logic [DATA_W - 1:0] sr [i-1:0];
                always_ff @( posedge clk or posedge rst ) begin : g_skew
                    if (rst || clr) begin 
                        for (int  k = 0; k < i; k++) sr[k] <= '0;
                    end
                    else if(en) begin
                        sr[0] <= data_in [i];
                        for (int k = 1; k < i; k++) begin
                            sr[k] <= sr[k-1]; 
                        end   
                    end   
                end

                assign data_out [i] = sr[i-1];
            end
                  
        end
endgenerate



endmodule;