`timescale 1ns / 1ps

// tb_top: N=16 MNIST inference regression for the current `top` (784 -> 150 -> 10).
//
// Verifies:
//   1) the four preload banks actually loaded from the .mem files ($readmemh)
//      by hierarchical readback of known words (X => the .mem files were not
//      found in the simulator working directory);
//   2) a full run completes (done) and `digit` matches the pure-Python
//      reference computed from data/fc*_*.bin + data/mem_files/image.mem.
//
// EXPECTED_DIGIT = 3 for the placeholder image.mem (digit "5"). If image.mem is
// regenerated, recompute the reference (data/mem_files) and update this value.

module tb_top;

    logic clk, rst, start, done;
    logic [3:0] digit;
    int fail = 0;

    localparam logic [3:0] EXPECTED_DIGIT = 4'd3;

    top dut (
        .clk   (clk),
        .rst   (rst),
        .start (start),
        .done  (done),
        .digit (digit)
    );

    always #5 clk = ~clk;

    // hierarchical readback of a preloaded word; fails on X or mismatch
    task automatic check_word(string name, input logic [127:0] got, input logic [127:0] exp);
        if ($isunknown(got)) begin
            $display("FAIL: %s = X -> .mem file not loaded (check sim working dir)", name);
            fail = 1;
        end else if (got !== exp) begin
            $display("FAIL: %s = %h exp %h", name, got, exp);
            fail = 1;
        end else begin
            $display("OK  : %s = %h", name, got);
        end
    endtask

    initial begin
        clk = 0; rst = 1; start = 0;

        // ---- 1) mem-load proof (banks are static after $readmemh) ----
        repeat (3) @(posedge clk);
        @(negedge clk); rst = 0;

        check_word("weights[0]   ", dut.u_weights.mem[0],
                   128'hfdfd030304fffd0203fdfc0102fdfd00);
        check_word("weights[7840]", dut.u_weights.mem[7840],
                   128'h000000000000150ed0eb23360abd04c5);
        check_word("bias[0]      ", 128'(dut.u_bias.mem[0]),
                   128'h0000000000000000000000000000075b);
        check_word("bias[160]    ", 128'(dut.u_bias.mem[160]),
                   128'h000000000000000000000000ffffffa9);
        check_word("instr[0]     ", 128'(dut.u_instr.mem[0]),
                   128'h00000000000000000000000000000000);
        check_word("instr[31]    ", 128'(dut.u_instr.mem[31]),
                   128'h0000000000000000000000096f500009);
        check_word("instr[32]    ", 128'(dut.u_instr.mem[32]),
                   128'h00000000000000000000000000140002);
        check_word("image[100]   ", dut.u_image.mem[100],
                   128'h7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f);

        if (fail) begin
            $display("ABORT: preload banks did not load; put the 4 .mem files in the");
            $display("       xsim working directory and rerun.");
            $finish;
        end
        $display("PRELOAD OK: all banks initialized from .mem files");

        // ---- 2) run one inference ----
        @(negedge clk); start = 1;
        @(negedge clk); start = 0;

        fork
            begin : timeout_t
                repeat (20000) @(posedge clk);
                $display("TIMEOUT waiting for done");
                fail = 1;
                disable check_t;
            end
            begin : check_t
                wait (done == 1);
                @(posedge clk);   // let digit settle past the done edge

                if (digit === EXPECTED_DIGIT) begin
                    $display("INFERENCE OK : digit = %0d (expected %0d)", digit, EXPECTED_DIGIT);
                end else begin
                    $display("FAIL: digit = %0d exp %0d (recompute EXPECTED_DIGIT if image.mem changed)",
                             digit, EXPECTED_DIGIT);
                    fail = 1;
                end
                disable timeout_t;
            end
        join

        if (fail) $display("TEST FAILED");
        else      $display("TEST PASSED");
        $finish;
    end

endmodule