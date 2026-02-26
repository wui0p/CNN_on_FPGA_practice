`timescale 1ns / 1ps

module tb_classifier;

    // =========================================================================
    // 1. CONFIGURATION
    // =========================================================================
    parameter string PATH = "D:/Senior/CNN_test/data/";
    
    // Input Size: 196 total values in golden_b4.hex
    // Split into 4 channels -> 49 values per channel
    parameter int INPUT_SIZE_TOTAL = 196;
    parameter int CH_SIZE = 49; 

    // =========================================================================
    // 2. SIGNALS
    // =========================================================================
    logic clk, reset;

    // --- DUT Inputs ---
    logic signed [15:0] pixel_in [0:3];
    logic valid_in;

    // --- Weights & Biases ---
    logic signed [15:0] weight [0:9][0:3][0:48];
    logic signed [31:0] bias [0:9];

    // --- DUT Outputs ---
    logic [3:0] result;
    logic done;

    // --- Memories ---
    // Stores the flat data from golden_b4.hex (196 lines)
    logic signed [15:0] img_mem [0:INPUT_SIZE_TOTAL-1]; 

    // --- Counters & Variables ---
    integer in_cnt;
    integer i;
    integer timeout;

    // =========================================================================
    // 3. DUT INSTANTIATION
    // =========================================================================
    classifier dut (
        .clk(clk),
        .reset(reset),
        .pixel_in(pixel_in),
        .valid_in(valid_in),
        .weight(weight),
        .bias(bias),
        .result(result),
        .done(done)
    );

    // Clock Generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // =========================================================================
    // 4. STIMULUS
    // =========================================================================
    initial begin
        // --- A. LOAD FILES ---
        $display("[%0t] Loading Files from %s...", $time, PATH);

        // 1. Load Input Data (196 values)
        $readmemh({PATH, "golden_b4.hex"}, img_mem);

        // 2. Load Classifier Weights & Biases
        $readmemh({PATH, "classifier_w.hex"}, weight);
        $readmemh({PATH, "classifier_b.hex"}, bias);

        // --- B. INITIALIZE ---
        reset = 1;
        valid_in = 0;
        in_cnt = 0;
        for(i=0; i<4; i++) pixel_in[i] = 0;

        #100; reset = 0; #20;

        $display("[%0t] Starting Classifier Inference...", $time);

        // --- C. FEED DATA ---
        // The classifier expects data for 49 cycles (indices 0 to 48).
        // We map the flat img_mem (196 values) into 4 parallel channels.
        // Assumption: File is organized Planar (Ch0 first, then Ch1, etc.)
        // Ch0: indices 0-48
        // Ch1: indices 49-97
        // Ch2: indices 98-146
        // Ch3: indices 147-195
        
        while (in_cnt < CH_SIZE) begin
            @(negedge clk);
            valid_in = 1;

            pixel_in[0] = img_mem[in_cnt];               // Channel 0
            pixel_in[1] = img_mem[in_cnt + CH_SIZE];     // Channel 1
            pixel_in[2] = img_mem[in_cnt + CH_SIZE*2];   // Channel 2
            pixel_in[3] = img_mem[in_cnt + CH_SIZE*3];   // Channel 3
            
            in_cnt++;
        end

        // Stop sending
        @(negedge clk);
        valid_in = 0;
        for(i=0; i<4; i++) pixel_in[i] = 0;

        // --- D. WAIT FOR OUTPUT ---
        $display("[%0t] Inputs finished. Waiting for classification...", $time);

        timeout = 0;
        while (!done && timeout < 1000) begin
            @(posedge clk);
            timeout++;
        end

        if (timeout >= 1000) begin
            $display("\n[ERROR] Simulation Timed Out waiting for done signal!");
            $finish;
        end

        // Wait one extra cycle to latch result if needed
        @(posedge clk);

        // --- E. FINAL REPORT ---
        $display("\n=======================================");
        $display("       CLASSIFIER RESULTS              ");
        $display("=======================================");
        $display(" Final Class Result : %0d", result);
        $display("=======================================\n");

        $finish;
    end

endmodule