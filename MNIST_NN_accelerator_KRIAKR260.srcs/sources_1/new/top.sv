`timescale 1ns/1ps

// MNIST 784 -> 150 -> 10 MLP on KRIA KR260 PL.
// 5 BRAM banks + pc + cu + NxN systolic array + post + argmax.
// Banks preloaded via $readmemh INIT_FILE (no load port).

module top #(
    parameter int N = 16,
    parameter string INIT_WEIGHTS = "weights.mem",
    parameter string INIT_BIAS    = "bias.mem",
    parameter string INIT_IMAGE   = "image.mem",
    parameter string INIT_INSTR   = "instr.mem"
)(
    input  logic clk, rst,
    input  logic start,
    output logic done,
    output logic [3:0] digit
);

    localparam int DATA_W = 8;
    localparam int ACC_W  = 32;
    localparam int WORD_W = N * DATA_W;      // 128 for N=16

    // ---------- banks ----------
    logic [63:0]       instr_out;
    logic [WORD_W-1:0] weights_out, image_out, act_out;
    logic [ACC_W-1:0]  bias_out;

    logic [6:0]  pc_addr;
    logic [9:0]  cu_image_addr, cu_act_addr;
    logic [12:0] cu_weights_addr;
    logic [7:0]  cu_bias_addr;
    logic        pc_clr, pc_inc;
    logic        act_we, feed_en, feed_sel, acc_clr, skew_clr;
    logic        post_qen, post_relu, logit_we, post_valid, a_sel;
    logic [4:0]  post_shift;
    logic [3:0]  post_j;
    logic [127:0] post_act_data;

    bram #(.DATA_W(64),    .ADDR_W(7),  .INIT_FILE(INIT_INSTR))   u_instr (
        .clk(clk), .we(1'b0), .addr(pc_addr),        .data_in('0), .data_out(instr_out)
    );
    bram #(.DATA_W(WORD_W), .ADDR_W(13), .INIT_FILE(INIT_WEIGHTS)) u_weights (
        .clk(clk), .we(1'b0), .addr(cu_weights_addr), .data_in('0), .data_out(weights_out)
    );
    bram #(.DATA_W(ACC_W), .ADDR_W(8),  .INIT_FILE(INIT_BIAS))   u_bias (
        .clk(clk), .we(1'b0), .addr(cu_bias_addr),    .data_in('0), .data_out(bias_out)
    );
    bram #(.DATA_W(WORD_W), .ADDR_W(10), .INIT_FILE(INIT_IMAGE)) u_image (
        .clk(clk), .we(1'b0), .addr(cu_image_addr),   .data_in('0), .data_out(image_out)
    );
    bram #(.DATA_W(WORD_W), .ADDR_W(8),  .INIT_FILE(""))         u_act (
        .clk(clk), .we(act_we), .addr(cu_act_addr[7:0]),
        .data_in(post_act_data), .data_out(act_out)
    );

    // ---------- control: pc + cu ----------
    pc #(.PC_W(7)) u_pc (
        .clk(clk), .rst(rst), .clr(pc_clr), .inc(pc_inc), .pc_addr(pc_addr)
    );

    cu #(.N(N), .A_W(10), .B_W(13), .BIAS_W(8), .T_W(13), .INSTR_W(64)) u_cu (
        .clk(clk), .rst(rst), .start(start), .instr(instr_out),
        .pc_clr(pc_clr), .pc_inc(pc_inc),
        .image_addr(cu_image_addr), .act_addr(cu_act_addr),
        .weights_addr(cu_weights_addr), .bias_addr(cu_bias_addr),
        .act_we(act_we),
        .feed_en(feed_en), .feed_sel(feed_sel),
        .acc_clr(acc_clr), .skew_clr(skew_clr),
        .post_qen(post_qen), .post_relu(post_relu), .post_shift(post_shift),
        .post_j(post_j), .post_valid(post_valid), .a_sel(a_sel),
        .logit_we(logit_we), .done(done)
    );

    // ---------- A/B feed path ----------
    logic [WORD_W-1:0] a_word;
    logic [DATA_W-1:0] skew_a_in [N-1:0], skew_b_in [N-1:0];
    logic [DATA_W-1:0] a_col [N-1:0], b_row [N-1:0];
    logic [ACC_W-1:0]  acc_out [N-1:0][N-1:0];

    // layer 1 MAC: image bank; layer 2 MAC: act bank (hidden activations)
    assign a_word = a_sel ? act_out : image_out;

    genvar i;
    generate
        for (i = 0; i < N; i++) begin : g_feed
            assign skew_a_in[i] = feed_sel ? a_word[8*i +: 8] : '0;
            assign skew_b_in[i] = feed_sel ? weights_out[8*i +: 8] : '0;
        end
    endgenerate

    skew_buffer #(.DATA_W(DATA_W), .N(N)) u_skew_a (
        .clk(clk), .rst(rst), .clr(skew_clr), .en(feed_en),
        .data_in(skew_a_in), .data_out(a_col)
    );
    skew_buffer #(.DATA_W(DATA_W), .N(N)) u_skew_b (
        .clk(clk), .rst(rst), .clr(skew_clr), .en(feed_en),
        .data_in(skew_b_in), .data_out(b_row)
    );

    systolic #(.DATA_W(DATA_W), .ACC_W(ACC_W), .N(N)) u_systolic (
        .clk(clk), .rst(rst), .acc_clr(acc_clr),
        .a_col(a_col), .b_row(b_row), .acc_out(acc_out)
    );

    // ---------- post + argmax ----------
    post #(.N(N), .ACC_W(ACC_W), .DATA_W(DATA_W)) u_post (
        .clk(clk), .rst(rst), .acc(acc_out), .bias(bias_out),
        .qen(post_qen), .relu(post_relu), .shift(post_shift),
        .j(post_j), .valid(post_valid), .act_data(post_act_data)
    );

    argmax #(.ACC_W(ACC_W)) u_argmax (
        .clk(clk), .rst(rst), .start(start),
        .logit_we(logit_we), .data_in(post_act_data), .digit(digit)
    );

endmodule
