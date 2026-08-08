`timescale  1ns/1ps

module sequencer #(
    parameter int DATA_W = 8,
    parameter int ACC_W  = 16,
    parameter int N      = 4,
    parameter int ADDR_W = 8,
    localparam int WORD_W     = N * ACC_W,
    localparam int T_FEED_END = N,
    localparam int T_COMP_END = 3*N - 2,
    localparam int T_WB_START = 3*N - 1,
    localparam int T_WB_END   = 4*N - 2,
    localparam int T_W        = $clog2(T_WB_END + 1)
)(
    input  logic               clk, rst,
    input  logic               start,
    input  logic [ACC_W-1:0]   acc [N-1:0][N-1:0],
    output logic               feed_en,
    output logic               feed_sel,
    output logic               we_a, we_b,
    output logic [ADDR_W-1:0]  addr_a, addr_b,
    output logic [WORD_W-1:0]  din_a,
    output logic               done
);

typedef enum logic [2:0] { IDLE, FEED, COMPUTE, WRITEBACK, DONE } state_t;
state_t state, next_state;

logic [T_W-1:0] t;

always_ff @(posedge clk or posedge rst) begin
    if (rst) begin
        state <= IDLE;
        t     <= '0;
    end else begin
        state <= next_state;
        t     <= (state == IDLE) ? '0 : t + 1'b1;
    end
end

always_comb begin
    next_state = state;
    case (state)
        IDLE:      if (start)            next_state = FEED;
        FEED:      if (t == T_FEED_END)  next_state = COMPUTE;
        COMPUTE:   if (t == T_COMP_END)  next_state = WRITEBACK;
        WRITEBACK: if (t == T_WB_END)    next_state = DONE;
        DONE:                            next_state = IDLE;
    endcase
end

logic [T_W-1:0] wb_row;
assign wb_row = t - T_WB_START;

assign feed_en  = (state == FEED)   ? (t >= 1) : (state == COMPUTE);
assign feed_sel = (state == FEED);
assign we_a     = (state == WRITEBACK);
assign we_b     = 1'b0;
assign done     = (state == DONE);

assign addr_a = (state == FEED)      ? (t < N ? t : N-1) :
                (state == WRITEBACK) ? (N + wb_row)       : '0;
assign addr_b = (state == FEED)      ? (t < N ? t : N-1) : '0;

always_comb begin
    din_a = '0;
    if (state == WRITEBACK) begin
        for (int j = 0; j < N; j++) begin
            din_a[(j*ACC_W) +: ACC_W] = acc[wb_row][j];
        end
    end
end

endmodule