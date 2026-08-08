`timescale  1ns/1ps

module systolic #(
    parameter int DATA_W = 8,
    parameter int ACC_W = 16,
    parameter int N = 4 // Number of PEs
)(
    input logic clk, rst,
    input logic [DATA_W-1:0] a_col [N-1:0], b_row [N-1:0],
    output logic [ACC_W-1:0] acc_out [N-1:0][N-1:0]
);

logic [DATA_W-1:0] a_bus [N-1:0][N:0];
logic [DATA_W-1:0] b_bus [N:0][N-1:0];



// structural modeling  

generate 
    genvar i, j ;
    for (i = 0 ; i < N; i++ ) begin : gen_row 
        for (j = 0 ; j < N; j++ ) begin : gen_col
            pmac uut_pmac (
                .clk(clk),
                .rst(rst),
                .a_in(a_bus[i][j]),
                .b_in(b_bus[i][j]),
                .a_out(a_bus[i][j+1]),
                .b_out(b_bus[i+1][j]),
                .acc_out(acc_out[i][j])
            );
        end 
    end  
endgenerate

// feed input into systolic array 
genvar k;
for ( k = 0 ; k < N; k++ ) begin : feed_input
    assign a_bus[k][0] = a_col[k];
    assign b_bus[0][k] = b_row[k];
end


// sequential (no need as pe handles sequential logic)
endmodule;