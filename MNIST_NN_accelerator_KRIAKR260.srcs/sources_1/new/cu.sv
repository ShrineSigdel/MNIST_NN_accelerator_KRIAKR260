`timescale 1ns/1ps

/*
    Always remember BRAM read is two cycles, first cycle is address, second cycle is data out.
*/

module cu #(
    parameter int N = 16,
    parameter int A_W = 10, // A bram width
    parameter int B_W = 13, // B bram width
    parameter int BIAS_W = 8, // BIAS bram width
    parameter int T_W = 13, // Run counter width ( K max )
    parameter int INSTR_W = 64  // Instruction width
)(
    input logic clk, rst, 
    input logic start, 
    input logic [INSTR_W-1:0] instr,


    // -- to pc.sv
    output logic pc_clr,          // 1 cycle on start -> pc = 0
    output logic pc_inc,          // during FETCH1 -> pc++

    // -- per bram addresses
    output logic [A_W-1:0] image_addr,
    output logic [A_W-1:0] act_addr,
    output logic [B_W-1:0] weights_addr,
    output logic [BIAS_W-1:0] bias_addr,
    output logic act_we, // act_we = 1 => writing activation else writing image

    // -- array / skew buffer control
    output logic feed_en, // feed_en = 1 => shifting data into skew buffers 
    output logic feed_sel, // feed_sel = 1 => feeding systolic with data when 1 ( 0 during drain )
    output logic acc_clr,
    output logic skew_clr,

    // -- decoded post signals / to post.sv
    output logic post_qen , // quantization enable
    output logic post_relu , 
    output logic [4:0] post_shift,
    output logic logit_we,  // QEN = 0, word-write pulse -> argmax

    // -- done signal
    output logic done
);

// instructions 
    localparam logic [2:0] OP_CLEAR = 3'b000;
    localparam logic [2:0] OP_MAC   = 3'b001;
    localparam logic [2:0] OP_POST  = 3'b010;
    localparam logic [2:0] OP_DONE  = 3'b011;
    localparam logic [2:0] OP_NOP   = 3'b100;

/*
FETCH1 and FETCH2 are used to fetch the instruction from the instruction memory
because the instruction memory is a BRAM and has a 2-cycle read latency.
*/

    typedef enum logic [2:0] { IDLE, FETCH1, FETCH2, RUN, DONE_ST } state_t;

    state_t state, next_state;

    logic [T_W-1:0] run_cnt; // run_cnt = cycle counter for RUN state
    logic [T_W-1:0] run_cnt_end;
    logic [INSTR_W-1:0] ir_reg;

    // instruction decode
    logic [2:0]  opcode;
    logic        a_sel, qen, relu;
    logic [A_W-1:0]  a_base;
    logic [B_W-1:0]  b_base;
    logic [11:0] k_cnt;
    logic [7:0]  dst_base, bias_base;
    logic [4:0]  shift;
    logic [5:0]  wait_cycles;

    assign opcode      = ir_reg[2:0];
    assign a_sel       = ir_reg[3];
    assign a_base      = ir_reg[14:5];
    assign b_base      = ir_reg[27:15];
    assign k_cnt       = ir_reg[39:28];
    assign qen         = ir_reg[3];
    assign dst_base    = ir_reg[12:5];
    assign bias_base   = ir_reg[20:13];
    assign shift       = ir_reg[25:21];
    assign relu        = ir_reg[26];
    assign wait_cycles = ir_reg[10:5];

    // state machine 

// ---- FSM (run-to-completion) ----
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state  <= IDLE;
            run_cnt <= '0;
            ir_reg <= '0;
        end else begin
            case (state)
                IDLE:    if (start) state <= FETCH1;
                FETCH1:  state <= FETCH2;
                FETCH2: begin
                    ir_reg <= instr;                
                    run_cnt <= '0;
                    if (instr[2:0] == OP_DONE)                             state <= DONE_ST;
                    else if (instr[2:0] == OP_NOP && instr[10:5] == 6'd0) state <= FETCH1;
                    else                                                    state <= RUN;
                end
                RUN:
                    if (run_cnt == run_cnt_end) begin
                        run_cnt <= '0;
                        state <= FETCH1;
                    end else begin
                        run_cnt <= run_cnt + 1'b1;
                    end
                DONE_ST: if (start) state <= IDLE;
            endcase
        end
    end


// per-instruction run counter
    
    always_comb begin
        case (opcode)
            OP_MAC:   run_cnt_end = k_cnt + 2*N - 2 ; // K + 2N - 1 cycles
            OP_POST:  run_cnt_end = 2*N - 1; // 2N cycles
            OP_NOP:   run_cnt_end = wait_cycles - 1'b1; // wait cycles
            default:  run_cnt_end = '0; // OP_CLEAR, OP_DONE = 1 cycle
        endcase
    end


// run-time helpers ( during run for address calculation )
logic [T_W-1:0] stream;   // min(t, K-1), saturated during drain 
logic [A_W-1:0] a_walk;
logic [B_W-1:0] b_walk;
logic [3:0]     post_j;


assign stream = (run_cnt < k_cnt) ? run_cnt : k_cnt - 1'b1;
assign a_walk = a_base + stream ;
assign b_walk = b_base + stream ;
assign post_j = run_cnt[4:1]; // Todo : to be understood later when implementing post.sv 


// -- pc control -- 
assign pc_clr = (state == IDLE && start) ;
assign pc_inc = (state == FETCH1) ;


 // ---- output generation ----
    always_comb begin
        acc_clr      = 1'b0;
        skew_clr     = 1'b0;
        feed_en      = 1'b0;
        feed_sel     = 1'b0;
        act_we       = 1'b0;
        logit_we     = 1'b0;
        image_addr   = '0;
        act_addr     = '0;
        weights_addr = '0;
        bias_addr    = '0;
        post_qen     = 1'b0;
        post_relu    = 1'b0;
        post_shift   = '0;

        unique case (state)
            RUN: begin
                unique case (opcode)
                    OP_CLEAR: begin
                        acc_clr  = 1'b1;
                        skew_clr = 1'b1;
                    end
                    OP_MAC: begin
                        feed_en      = (run_cnt >= 1);
                        feed_sel     = (run_cnt >= 1) && (run_cnt <= k_cnt); // i.e 0 during drain
                        weights_addr = b_walk;
                        if (a_sel) act_addr = a_walk;
                        else       image_addr = a_walk;
                    end
                    OP_POST: begin // Todo: to be understood later when implementing post.sv
                        post_qen   = qen;
                        post_relu  = relu;
                        post_shift = shift;
                        act_addr  = qen ? (dst_base + post_j)
                                            : (dst_base + (post_j >> 2));
                        if (run_cnt[0] == 1'b0) begin
                            bias_addr = bias_base + post_j;
                            
                        end else begin
                            act_we   = qen ? 1'b1 : (post_j[1:0] == 2'b11);
                            logit_we = ~qen && act_we;
                        end
                    end
                    default: ; // NOP: idle
                endcase
            end
            default: ; // IDLE / FETCH1 / FETCH2 / DONE_ST
        endcase
    end

assign done = (state == DONE_ST);

endmodule;