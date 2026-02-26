`timescale 1ns / 1ps

module tb_layer_1;

    // =========================================================================
    // 1. PARAMETERS & PATHS
    // =========================================================================
    parameter string PATH = "D:/Senior/CNN_test/data/";
    
    // =========================================================================
    // 2. SIGNALS
    // =========================================================================
    logic clk, reset;
    
    // Inputs to DUT
    logic signed [15:0] pixel_in;
    logic valid_in;
    
    // Weights & Biases
    logic signed [15:0] weight_b1 [0:3][0:8];
    logic signed [31:0] bias_b1 [0:3];

    // Outputs from DUT
    logic ready_out;       
    logic signed [15:0] pixel_out [0:3];
    logic [3:0] valid_out;

    // =========================================================================
    // 3. MEMORIES & VARIABLES
    // =========================================================================
    logic signed [15:0] img_mem [0:783]; 
    logic signed [15:0] gold_mem_total [0:3135]; 

    integer f_out;
    integer in_count = 0;
    integer timeout = 0;
    integer i;

    // --- VERIFICATION COUNTERS ---
    integer out_cnt [0:3];   
    integer err_cnt [0:3];   
    // REMOVED: integer skip_lat [0:3];  <-- No longer needed
    
    // Final Totals
    integer total_checked = 0;
    integer total_errors = 0;
    
    // Flag for smart stop
    logic all_done;

    // =========================================================================
    // 4. DUT INSTANTIATION
    // =========================================================================
    layer_1 dut (
        .clk(clk),
        .reset(reset),
        .pixel_in(pixel_in),
        .valid_in(valid_in),
        .weight_b1(weight_b1), 
        .bias_b1(bias_b1),
        .ready(ready_out),
        .pixel_out(pixel_out),
        .valid_out(valid_out)
    );

    // Clock Generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // =========================================================================
    // 5. STIMULUS & REPORTING
    // =========================================================================
    initial begin
        // --- 1. Load Files ---
        $display("Loading files from %s...", PATH);
        $readmemh({PATH, "test_image_0.hex"}, img_mem);
        $readmemh({PATH, "conv_b1_w.hex"}, weight_b1);
        $readmemh({PATH, "conv_b1_b.hex"}, bias_b1);
        $readmemh({PATH, "golden_b1.hex"}, gold_mem_total); 

        f_out = $fopen({PATH, "hw_output_log.txt"}, "w");

        // --- 2. Initialize Counters ---
        for (i=0; i<4; i=i+1) begin
            out_cnt[i] = 0;
            err_cnt[i] = 0;
            // REMOVED: skip_lat initialization
        end

        // --- 3. Reset ---
        reset = 1; valid_in = 0; pixel_in = 0;
        #100; reset = 0; #20;

        $display("--- SIMULATION STARTED ---");

        // --- 4. Send Image ---
        while (in_count < 784) begin
            @(negedge clk); 
            if (ready_out) begin
                valid_in = 1; 
                pixel_in = img_mem[in_count]; 
                in_count = in_count + 1;
            end else begin
                valid_in = 0; 
            end
        end

        // --- 5. SMART WAIT LOOP ---
        $display("Image sent. Waiting for last pixel...");
        timeout = 0;
        all_done = 0;

        while (!all_done && timeout < 5000) begin
            @(negedge clk);
            
            // Flush pipeline
            if (ready_out) begin 
                valid_in = 1; 
                pixel_in = 0; 
            end else begin 
                valid_in = 0;
            end
            
            // Check completion
            if (out_cnt[0] >= 784 && out_cnt[1] >= 784 && 
                out_cnt[2] >= 784 && out_cnt[3] >= 784) begin
                all_done = 1;
            end

            timeout = timeout + 1;
        end
        
        // --- 6. Extra Delay ---
        repeat(10) @(posedge clk);
        
        // --- 7. FINAL REPORTING ---
        $display("\n==================================================");
        $display("             FINAL SIMULATION REPORT              ");
        $display("==================================================");
        
        total_checked = 0;
        total_errors = 0;
        
        for (i=0; i<4; i=i+1) begin
            total_checked = total_checked + out_cnt[i];
            total_errors = total_errors + err_cnt[i];
            $display("CHANNEL %0d: Checked %0d | Errors: %0d", i+1, out_cnt[i], err_cnt[i]);
        end

        $display("--------------------------------------------------");
        $display("TOTAL PIXELS CHECKED : %0d / 3136", total_checked);
        $display("TOTAL MISMATCHES     : %0d", total_errors);
        $display("--------------------------------------------------");

        if (total_errors == 0 && total_checked == 3136) begin
            $display("\n  [ SUCCESS ]  HARDWARE MATCHES GOLDEN MODEL PERFECTLY \n");
        end else begin
            $display("\n  [ FAILURE ]  FOUND %0d ERRORS \n", total_errors);
        end
        $display("==================================================");

        $fclose(f_out);
        $finish;
    end

    // =========================================================================
    // 6. VERIFICATION: 4 PARALLEL CHECKERS (FIXED)
    // =========================================================================
    genvar k;
    generate
        for (k=0; k<4; k=k+1) begin : checkers
            always @(posedge clk) begin
                if (valid_out[k]) begin
                    // --- REMOVED: if (skip_lat) check ---
                    // Now we check immediately on the first valid_out
                    
                    if (out_cnt[k] < 784) begin
                        logic signed [15:0] current_pixel;
                        logic signed [15:0] golden_val;
                        integer g_idx;

                        current_pixel = pixel_out[k];
                        g_idx = (k * 784) + out_cnt[k];
                        golden_val = gold_mem_total[g_idx];

                        if (current_pixel !== golden_val) begin
                            $display("ERROR CH%0d [Px %0d] | HW: %h | GOLD: %h", 
                                     k+1, out_cnt[k], current_pixel, golden_val);
                            err_cnt[k] = err_cnt[k] + 1;
                        end
                        out_cnt[k] = out_cnt[k] + 1;
                    end
                end
            end
        end
    endgenerate

endmodule