`timescale 1ns/1ps

module post #(
    parameter int N      = 16,
    parameter int ACC_W  = 32,
    parameter int DATA_W = 8
)(
    input  logic clk, rst,
    input  logic [ACC_W-1:0] acc [N-1:0][N-1:0],  // systolic acc_out[i][j]
    input  logic [ACC_W-1:0]               bias,  // bias bank dout for neuron j
    input  logic                           qen,   // 1 = hidden (int8), 0 = logits (int32)
    input  logic                           relu,
    input  logic [4:0]                     shift,
    input  logic [3:0]                     j,     // neuron index 0..15 (from cu: run_cnt[4:1])
    input  logic                           valid, // cu strobe: 1 on each POST write phase (odd run_cnt)
    output logic [127:0]                   act_data
);

    logic signed [ACC_W-1:0] s [N-1:0];          // transformed per-lane result
    logic signed [DATA_W-1:0] v [N-1:0];         // clamped int8 (qen=1 only)

    // 16-lane transform: acc + bias -> relu? -> arithmetic >>shift -> clamp int8
    always_comb begin
        for (int i = 0; i < N; i++) begin
            s[i] = $signed(acc[i][j]) + $signed(bias);
            if (relu && s[i] < 0) s[i] = '0;
            s[i] = s[i] >>> shift;

            //safety net 
            if      (s[i] >  127)  v[i] = 127;
            else if (s[i] < -128)  v[i] = -128;
            else                   v[i] = s[i][7:0];
        end
    end

    // QEN=0 path: hold the last 4 neurons' lane-0 logit until a full word is ready
    logic [ACC_W-1:0] logit_buf [0:3];
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            for (int k = 0; k < 4; k++) logit_buf[k] <= '0;
        end else if (valid && ~qen) begin
            logit_buf[j[1:0]] <= s[0];           // lane-0 logit for class j (M=1)
        end
    end

    // packing: qen=1 -> 16 x int8 (lane i at byte i);
    //          qen=0 -> 4 x int32 (word q = classes 4q..4q+3, class 4q+k at bits [32k+:32])
    always_comb begin
        if (qen) begin
            act_data = { v[15], v[14], v[13], v[12],
                         v[11], v[10], v[9],  v[8],
                         v[7],  v[6],  v[5],  v[4],
                         v[3],  v[2],  v[1],  v[0] };
        end else begin
            act_data = { s[0], logit_buf[2], logit_buf[1], logit_buf[0] };
        end
    end

endmodule