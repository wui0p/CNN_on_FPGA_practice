`timescale 1ns / 1ps

module tb_layer_4;

    // =========================================================================
    // 1. CONFIGURATION
    // =========================================================================
    parameter string PATH = "D:/Senior/CNN_test/data/";
    
    // Input: Layer 1 takes 28x28 image
    parameter int IMG_IN_SIZE  = 784; 
    
    // Output: Layer 4 has Max Pooling (14x14 -> 7x7)
    // 7 * 7 = 49 pixels per channel
    parameter int IMG_OUT_SIZE = 49; 
    
    // =========================================================================
    // 2. SIGNALS
    // =========================================================================
    logic clk, reset;
    
    // --- DUT Inputs ---
    logic signed [15:0] pixel_in;
    logic valid_in;
    
    // --- Weights & Biases (All 4 Layers) ---
    logic signed [15:0] weight_b1 [0:3][0:8];
    logic signed [31:0] bias_b1 [0:3];
    
    logic signed [15:0] weight_b2 [0:3][0:3][0:8];
    logic signed [31:0] bias_b2 [0:3];
    
    logic signed [15:0] weight_b3 [0:3][0:3][0:8];
    logic signed [31:0] bias_b3 [0:3];
    
    logic signed [15:0] weight_b4 [0:3][0:3][0:8]; // New Layer 4
    logic signed [31:0] bias_b4 [0:3];             // New Layer 4

    // --- DUT Outputs ---
    logic signed [15:0] pixel_out [0:3];
    logic ready;
    logic [3:0] valid_out;

    // --- Memories ---
    logic signed [15:0] img_mem [0:783];      
    logic signed [15:0] gold_mem [0:195];     // 4 channels * 49 pixels = 196 total words

    // --- Counters ---
    integer in_cnt;
    integer out_cnt [0:3];
    integer err_cnt [0:3];
    integer i;
    integer total_errors;
    
    // =========================================================================
    // 3. DUT INSTANTIATION
    // =========================================================================
    // Ensure your top module is named conv_top_b4 !
    conv_top_b4 dut (
        .clk(clk),
        .reset(reset),
        
        .pixel_in(pixel_in),   
        .valid_in(valid_in),
        
        // Connect all weights
        .weight_b1(weight_b1), .bias_b1(bias_b1),
        .weight_b2(weight_b2), .bias_b2(bias_b2),
        .weight_b3(weight_b3), .bias_b3(bias_b3),
        .weight_b4(weight_b4), .bias_b4(bias_b4),
        
        .pixel_out(pixel_out),
        .ready(ready),
        .valid_out(valid_out)
    );

    // Clock
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
        
        // 1. Image
        $readmemh({PATH, "test_image_0.hex"}, img_mem);
        
        // 2. Weights/Biases
        $readmemh({PATH, "conv_b1_w.hex"}, weight_b1);
        $readmemh({PATH, "conv_b1_b.hex"}, bias_b1);
        
        $readmemh({PATH, "conv_b2_w.hex"}, weight_b2);
        $readmemh({PATH, "conv_b2_b.hex"}, bias_b2);
        
        $readmemh({PATH, "conv_b3_w.hex"}, weight_b3);
        $readmemh({PATH, "conv_b3_b.hex"}, bias_b3);
        
        // New L4 Files
        $readmemh({PATH, "conv_b4_w.hex"}, weight_b4);
        $readmemh({PATH, "conv_b4_b.hex"}, bias_b4);
        
        // 3. Golden Output (Layer 4 Result)
        $readmemh({PATH, "golden_b4.hex"}, gold_mem); 

        // --- B. INITIALIZE ---
        reset = 1;
        valid_in = 0;
        in_cnt = 0;
        pixel_in = 0;
        
        for(i=0; i<4; i++) begin
            out_cnt[i] = 0;
            err_cnt[i] = 0;
        end

        #100; reset = 0; #20;
        
        $display("[%0t] Starting Full 4-Layer Simulation...", $time);

        // --- C. FEED DATA ---
        while (in_cnt < IMG_IN_SIZE) begin
            @(negedge clk);
            if (ready) begin
                valid_in = 1;
                pixel_in = img_mem[in_cnt];
                in_cnt++;
            end else begin
                valid_in = 0;
            end
        end
        
        @(negedge clk);
        valid_in = 0;

        // --- D. WAIT FOR OUTPUTS ---
        $display("[%0t] Input done. Waiting for %0d outputs (7x7)...", $time, IMG_OUT_SIZE);
        
        fork
            begin : wait_done
                wait(out_cnt[0] == IMG_OUT_SIZE && out_cnt[1] == IMG_OUT_SIZE && 
                     out_cnt[2] == IMG_OUT_SIZE && out_cnt[3] == IMG_OUT_SIZE);
            end
            begin : timeout
                #500000; // Large timeout for deep pipeline
                $display("\n[ERROR] Simulation Timed Out!");
                $display("Status: Ch0:%d Ch1:%d Ch2:%d Ch3:%d", out_cnt[0], out_cnt[1], out_cnt[2], out_cnt[3]);
                $finish;
            end
        join_any
        disable timeout;

        // --- E. REPORT ---
        $display("\n=======================================");
        $display("       LAYER 4 RESULTS (7x7)           ");
        $display("=======================================");
        total_errors = 0;
        for(i=0; i<4; i++) begin
             total_errors += err_cnt[i];
             $display("CH%0d: Processed %0d | Errors: %0d", i, out_cnt[i], err_cnt[i]);
        end

        $display("---------------------------------------");
        if(total_errors == 0) 
            $display(" [SUCCESS] Hardware Matches Golden Model!");
        else 
            $display(" [FAILURE] Found %0d mismatches.", total_errors);
        $display("=======================================\n");
        
        $finish;
    end

    // =========================================================================
    // 5. CHECKER LOGIC
    // =========================================================================
    genvar ch;
    generate
        for(ch=0; ch<4; ch++) begin : checkers
            always @(posedge clk) begin
                if (valid_out[ch]) begin
                    if (out_cnt[ch] < IMG_OUT_SIZE) begin
                        logic signed [15:0] hw;
                        logic signed [15:0] gold;
                        
                        hw = pixel_out[ch];
                        // Golden packing: Ch0[0..48], Ch1[0..48]...
                        gold = gold_mem[(ch * IMG_OUT_SIZE) + out_cnt[ch]];

                        if (hw !== gold) begin
                            if (err_cnt[ch] < 20) begin 
                                $display("ERR CH%0d Px%0d | HW: %h | Gold: %h", 
                                         ch, out_cnt[ch], hw, gold);
                            end
                            err_cnt[ch]++;
                        end
                        out_cnt[ch]++;
                    end
                end
            end
        end
    endgenerate

endmodule