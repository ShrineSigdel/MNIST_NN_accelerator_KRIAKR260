`timescale  1ns/1ps

module top #(
    parameter int DATA_W = 8,
    parameter int ACC_W  = 32,
    parameter int N      = 4,
    parameter int ADDR_W = 8,
    localparam int WORD_W = N * ACC_W
)(
    input  logic clk, rst,
    input  logic load,
    input  logic start,
    input  logic load_we_a, load_we_b,
    input  logic [ADDR_W-1:0] load_addr_a, load_addr_b,
    input  logic [WORD_W-1:0] load_din_a, load_din_b,
    output logic        done,
    output logic [ACC_W-1:0] acc_out [N-1:0][N-1:0]
);

logic seq_we_a, seq_we_b, feed_en, feed_sel;
logic [ADDR_W-1:0] seq_addr_a, seq_addr_b;
logic [WORD_W-1:0] seq_din_a, bram_out_a, bram_out_b;
logic [DATA_W-1:0] skew_din_a [N-1:0], skew_din_b [N-1:0];
logic [DATA_W-1:0] a_col [N-1:0], b_row [N-1:0];
logic we_a, we_b;
logic [ADDR_W-1:0] addr_a, addr_b;
logic [WORD_W-1:0] din_a;

bram #(.DATA_W(WORD_W), .ADDR_W(ADDR_W)) u_bram (
    .clk_a(clk), .we_a(we_a), .addr_a(addr_a), .data_in_a(din_a), .data_out_a(bram_out_a),
    .clk_b(clk), .we_b(we_b), .addr_b(addr_b), .data_in_b(load_din_b), .data_out_b(bram_out_b)
);

assign we_a   = load ? load_we_a  : seq_we_a;
assign we_b   = load ? load_we_b  : seq_we_b;
assign addr_a = load ? load_addr_a : seq_addr_a;
assign addr_b = load ? load_addr_b : seq_addr_b;
assign din_a  = load ? load_din_a  : seq_din_a;

generate
    genvar i;
    for (i = 0; i < N; i++) begin : unpack
        assign skew_din_a[i] = (feed_en && feed_sel) ? bram_out_a[i*ACC_W +: DATA_W] : '0;
        assign skew_din_b[i] = (feed_en && feed_sel) ? bram_out_b[i*ACC_W +: DATA_W] : '0;
    end
endgenerate

skew_buffer #(.DATA_W(DATA_W), .N(N)) u_skew_a (
    .clk(clk), .rst(rst), .en(feed_en), .data_in(skew_din_a), .data_out(a_col)
);
skew_buffer #(.DATA_W(DATA_W), .N(N)) u_skew_b (
    .clk(clk), .rst(rst), .en(feed_en), .data_in(skew_din_b), .data_out(b_row)
);

systolic #(.DATA_W(DATA_W), .ACC_W(ACC_W), .N(N)) u_systolic (
    .clk(clk), .rst(rst), .a_col(a_col), .b_row(b_row), .acc_out(acc_out)
);

sequencer #(.DATA_W(DATA_W), .ACC_W(ACC_W), .N(N), .ADDR_W(ADDR_W)) u_seq (
    .clk(clk), .rst(rst), .start(start), .acc(acc_out),
    .feed_en(feed_en), .feed_sel(feed_sel),
    .we_a(seq_we_a), .we_b(seq_we_b),
    .addr_a(seq_addr_a), .addr_b(seq_addr_b),
    .din_a(seq_din_a), .done(done)
);

endmodule