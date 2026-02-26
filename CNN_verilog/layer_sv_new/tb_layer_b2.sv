`timescale 1ns / 1ps

module tb_layer_2;

    // =========================================================================
    // 1. CONFIGURATION
    // =========================================================================
    parameter string PATH = "D:/Senior/CNN_test/data/";
    
    // Input is now the original 28x28 image
    parameter int IMG_IN_SIZE  = 784; 
    
    // Output is the result of Layer 2 (14x14)
    parameter int IMG_OUT_SIZE = 196; 
    
    // =========================================================================
    // 2. SIGNALS
    // =========================================================================
    logic clk, reset;
    
    // --- DUT Inputs (Layer 1 Interface) ---
    logic signed [15:0] pixel_in;      // Single channel input for Layer 1
    logic valid_in;
    
    // --- Weights & Biases (Both Layers) ---
    logic signed [15:0] weight_b1 [0:3][0:8];
    logic signed [31:0] bias_b1 [0:3];
    logic signed [15:0] weight_b2 [0:3][0:3][0:8];
    logic signed [31:0] bias_b2 [0:3];

    // --- DUT Outputs (Layer 2 Interface) ---
    logic signed [15:0] pixel_out [0:3];
    logic ready;
    logic [3:0] valid_out;

    // --- Memories ---
    logic signed [15:0] img_mem [0:783];      // Input Image (0-255 grayscale)
    logic signed [15:0] gold_mem [0:783];     // Golden Output of Layer 2 (4ch * 196 = 784 total)

    // --- Counters ---
    integer in_cnt;
    integer out_cnt [0:3];
    integer err_cnt [0:3];
    integer i;
    integer total_errors;
    
    // =========================================================================
    // 3. DUT INSTANTIATION
    // =========================================================================
    conv_top_b2 dut (
        .clk(clk),
        .reset(reset),
        // Layer 1 Inputs
        .pixel_in(pixel_in),   
        .valid_in(valid_in),
        .weight_b1(weight_b1),
        .bias_b1(bias_b1),
        // Layer 2 Inputs
        .weight_b2(weight_b2),
        .bias_b2(bias_b2),
        // Outputs
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
        
        // 1. Input Image
        $readmemh({PATH, "test_image_0.hex"}, img_mem);
        
        // 2. Layer 1 Params
        $readmemh({PATH, "conv_b1_w.hex"}, weight_b1);
        $readmemh({PATH, "conv_b1_b.hex"}, bias_b1);
        
        // 3. Layer 2 Params
        $readmemh({PATH, "conv_b2_w.hex"}, weight_b2);
        $readmemh({PATH, "conv_b2_b.hex"}, bias_b2);
        
        // 4. Golden Output (Results of Layer 2)
        $readmemh({PATH, "golden_b2.hex"}, gold_mem); 

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
        
        $display("[%0t] Starting Combined Layer 1 + Layer 2 Simulation...", $time);

        // --- C. FEED DATA (Layer 1 Protocol) ---
        // We send 1 pixel at a time, respecting the 'ready' signal
        while (in_cnt < IMG_IN_SIZE) begin
            @(negedge clk);
            if (ready) begin
                valid_in = 1;
                pixel_in = img_mem[in_cnt];
                in_cnt++;
            end else begin
                // If DUT is not ready (FIFO full or processing), hold valid low
                valid_in = 0;
            end
        end
        
        // Stop sending
        @(negedge clk);
        valid_in = 0;

        // --- D. WAIT FOR OUTPUTS (Layer 2 Protocol) ---
        $display("[%0t] Inputs finished. Waiting for %0d outputs per channel...", $time, IMG_OUT_SIZE);
        
        fork
            begin : wait_done
                wait(out_cnt[0] == IMG_OUT_SIZE && out_cnt[1] == IMG_OUT_SIZE && 
                     out_cnt[2] == IMG_OUT_SIZE && out_cnt[3] == IMG_OUT_SIZE);
            end
            begin : timeout
                // Increased timeout because now we are running two layers
                #200000;
                $display("\n[ERROR] Simulation Timed Out!");
                $display("Status: Ch0:%d Ch1:%d Ch2:%d Ch3:%d", out_cnt[0], out_cnt[1], out_cnt[2], out_cnt[3]);
                $finish;
            end
        join_any
        disable timeout;

        // --- E. FINAL REPORT ---
        $display("\n=======================================");
        $display("       FULL PIPELINE RESULTS           ");
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
                        // Golden memory is usually packed sequentially:
                        // Ch0[0..195], Ch1[0..195], etc.
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