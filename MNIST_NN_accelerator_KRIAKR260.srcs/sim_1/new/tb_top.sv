`timescale 1ns / 1ps

module tb_top;

parameter int N = 4, DATA_W = 8, ACC_W = 32, ADDR_W = 8;
localparam int WORD_W = N * ACC_W;

logic clk, rst, load, start, done;
logic load_we_a, load_we_b;
logic [ADDR_W-1:0] load_addr_a, load_addr_b;
logic [WORD_W-1:0] load_din_a, load_din_b;
logic [ACC_W-1:0] acc_out [N-1:0][N-1:0];

int A [N][N], B [N][N], C [N][N];
int fail = 0;

top #(.DATA_W(DATA_W), .ACC_W(ACC_W), .N(N), .ADDR_W(ADDR_W)) dut (
    .clk(clk), .rst(rst), .load(load), .start(start),
    .load_we_a(load_we_a), .load_we_b(load_we_b),
    .load_addr_a(load_addr_a), .load_addr_b(load_addr_b),
    .load_din_a(load_din_a), .load_din_b(load_din_b),
    .done(done), .acc_out(acc_out)
);

always #5 clk = ~clk;

// per-cycle monitor of key signals (state != IDLE)
always @(posedge clk) begin
    if (!rst && dut.u_seq.state != 3'b000)
        $display("t=%0t state=%0d feed_en=%0b feed_sel=%0b addr_a=%0d addr_b=%0d | a_col={%0d %0d %0d %0d} b_row={%0d %0d %0d %0d}",
            $time, dut.u_seq.state, dut.feed_en, dut.feed_sel,
            dut.u_seq.addr_a, dut.u_seq.addr_b,
            dut.a_col[0], dut.a_col[1], dut.a_col[2], dut.a_col[3],
            dut.b_row[0], dut.b_row[1], dut.b_row[2], dut.b_row[3]);
end

// pack a full A column / B row / C row into a single WORD_W word
function automatic logic [WORD_W-1:0] pack_col(input int idx);
    logic [WORD_W-1:0] w;
    w = '0;
    for (int i = 0; i < N; i++)
        w[i*ACC_W +: DATA_W] = A[i][idx][DATA_W-1:0];
    return w;
endfunction

function automatic logic [WORD_W-1:0] pack_row(input int idx);
    logic [WORD_W-1:0] w;
    w = '0;
    for (int j = 0; j < N; j++)
        w[j*ACC_W +: DATA_W] = B[idx][j][DATA_W-1:0];
    return w;
endfunction

function automatic logic [WORD_W-1:0] pack_crow(input int r);
    logic [WORD_W-1:0] w;
    w = '0;
    for (int j = 0; j < N; j++)
        w[j*ACC_W +: ACC_W] = C[r][j][ACC_W-1:0];
    return w;
endfunction

initial begin
    clk = 0; rst = 1; load = 0; start = 0;
    load_we_a = 0; load_we_b = 0;
    load_addr_a = 0; load_addr_b = 0;
    load_din_a = 0; load_din_b = 0;

    for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) begin
        A[i][j] = (i == j) ? (i + 1) : 0;   // diagonal A
        B[i][j] = i + 1;                    // B: row i all = i+1
    end

    // expected C = A * B
    for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) begin
        C[i][j] = 0;
        for (int k = 0; k < N; k++) C[i][j] += A[i][k] * B[k][j];
    end

    // hold reset for 3 cycles, release race-free on a negedge
    repeat (3) @(posedge clk);
    @(negedge clk); rst = 0;

    // ---- Load A through port A: mem_a[k] = packed column k ----
    @(negedge clk); load = 1;
    for (int k = 0; k < N; k++) begin
        @(negedge clk);
        load_we_a = 1; load_addr_a = k; load_din_a = pack_col(k);
    end
    @(negedge clk); load_we_a = 0;

    // ---- Load B through port B: mem_b[k] = packed row k ----
    for (int k = 0; k < N; k++) begin
        @(negedge clk);
        load_we_b = 1; load_addr_b = k; load_din_b = pack_row(k);
    end
    @(negedge clk); load_we_b = 0;

    // ---- Pre-flight readback check (abort early if memory is empty) ----
    @(negedge clk); load_addr_a = 0; load_addr_b = 0;
    @(posedge clk);                 // data_out latches mem[0]
    @(negedge clk);                 // settled, race-free read
    if (dut.bram_out_a !== pack_col(0)) begin
        $display("MEM-A LOAD FAIL: bram_out_a=%h exp=%h", dut.bram_out_a, pack_col(0));
        fail = 1;
    end
    if (dut.bram_out_b !== pack_row(0)) begin
        $display("MEM-B LOAD FAIL: bram_out_b=%h exp=%h", dut.bram_out_b, pack_row(0));
        fail = 1;
    end
    if (fail) begin
        $display("ABORT: memory load did not take; fix load path first");
        $finish;
    end
    $display("MEMORY LOAD OK (bram_out_a=%h bram_out_b=%h)", dut.bram_out_a, dut.bram_out_b);

    // ---- Reset-state check (acc_out must be 0 before run) ----
    for (int i = 0; i < N; i++) for (int j = 0; j < N; j++)
        if (acc_out[i][j] !== '0) begin
            $display("RESET-FAIL acc_out[%0d][%0d] = %0d", i, j, acc_out[i][j]);
            fail = 1;
        end

    // ---- Run GEMM ----
    @(negedge clk); load = 0;
    @(negedge clk); start = 1;
    @(posedge clk);              // FSM samples start=1, enters FEED
    @(negedge clk); start = 0;

    fork
        begin : timeout_t
            repeat (200) @(posedge clk);
            $display("TIMEOUT waiting for done");
            fail = 1;
            disable check_t;
        end
        begin : check_t
            wait (done == 1);
            @(posedge clk);

            for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) begin
                if ($isunknown(acc_out[i][j])) begin
                    $display("X-DETECTED acc_out[%0d][%0d]", i, j);
                    fail = 1;
                end else if (acc_out[i][j] !== C[i][j][ACC_W-1:0]) begin
                    $display("MISMATCH [%0d][%0d] got %0d exp %0d", i, j, acc_out[i][j], C[i][j]);
                    fail = 1;
                end
            end

            // ---- C writeback readback via port A: addr N+r ----
            for (int r = 0; r < N; r++) begin
                @(negedge clk);
                load = 1; load_we_a = 0; load_addr_a = N + r;
                @(posedge clk);         // data_out latches mem_a[N+r]
                @(negedge clk);         // settled, race-free read
                if (dut.bram_out_a !== pack_crow(r)) begin
                    $display("WB-MISMATCH row %0d: got %h exp %h", r, dut.bram_out_a, pack_crow(r));
                    fail = 1;
                end
            end
            @(negedge clk); load = 0;

            disable timeout_t;
        end
    join

    if (fail) $display("TEST FAILED");
    else      $display("TEST PASSED");
    $finish;
end

endmodule
