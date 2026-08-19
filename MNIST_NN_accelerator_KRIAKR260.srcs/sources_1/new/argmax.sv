`timescale 1ns/1ps

module argmax #(
    parameter int ACC_W = 32
)(
    input  logic clk, rst,
    input  logic start,            // re-arm running max for a new inference
    input  logic logit_we,         // QEN=0 logit-word write strobe (cu.logit_we)
    input  logic [127:0] data_in,  // 4 x int32 logit word (post.act_data)
    output logic [3:0] digit       // winning class index (valid when done)
);

    // ---- class-enable mask per word (v1: classes 0-9 real) ----
    logic [1:0] word_cnt;
    logic [3:0] mask;

    always_comb begin
        unique case (word_cnt)
            2'd0, 2'd1: mask = 4'b1111;   // classes 0-3, 4-7: all real
            2'd2:       mask = 4'b0011;   // classes 8-9 real, 10-11 pad
            default:    mask = 4'b0000;   // classes 12-15: all pad
        endcase
    end

    // ---- best within the incoming word, over enabled slots ----
    logic signed [ACC_W-1:0] slot_val [0:3];
    logic signed [ACC_W-1:0] wval;
    logic [3:0] widx;

    assign slot_val[0] = $signed(data_in[31:0]);
    assign slot_val[1] = $signed(data_in[63:32]);
    assign slot_val[2] = $signed(data_in[95:64]);
    assign slot_val[3] = $signed(data_in[127:96]);

    always_comb begin
        wval = 32'sh8000_0000;            // INT32_MIN
        widx = '0;
        for (int k = 0; k < 4; k++) begin
            if (mask[k] && slot_val[k] > wval) begin
                wval = slot_val[k];
                widx = {word_cnt, k[1:0]}; // class index = word_cnt*4 + k
            end
        end
    end

    // ---- running best across the 4 words ----
    logic signed [ACC_W-1:0] best_val;
    logic [3:0] best_idx;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            best_val <= 32'sh8000_0000;
            best_idx <= '0;
            word_cnt <= '0;
        end else if (start) begin
            best_val <= 32'sh8000_0000;
            best_idx <= '0;
            word_cnt <= '0;
        end else if (logit_we) begin
            if (wval > best_val) begin
                best_val <= wval;
                best_idx <= widx;
            end
            word_cnt <= word_cnt + 1'b1;
        end
    end

    assign digit = best_idx;

endmodule