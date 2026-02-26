`timescale 1ns / 1ps

module tb_layer_b2;

    // =========================================================================
    // 1. CONFIGURATION
    // =========================================================================
    parameter string PATH = "D:/Senior/CNN_test/data/";
    
    parameter int IMG_IN_SIZE  = 784; // 28x28 input
    parameter int IMG_OUT_SIZE = 196; // 14x14 output
    
    // REMOVED: CHECK_DELAY parameter (No longer needed)

    // =========================================================================
    // 2. SIGNALS
    // =========================================================================
    logic clk, reset;
    
    // DUT Inputs
    logic signed [15:0] pixel_in [0:3];
    logic valid_in;
    
    // DUT Weights/Biases
    logic signed [15:0] weight_b2 [0:3][0:3][0:8];
    logic signed [31:0] bias_b2 [0:3];

    // DUT Outputs
    logic signed [15:0] pixel_out [0:3];
    logic ready;
    logic [3:0] valid_out;

    // REMOVED: Pipeline registers (valid_d1, valid_d2, valid_to_check)

    // Memories
    logic signed [15:0] input_mem [0:3135]; 
    logic signed [15:0] gold_mem [0:783];   

    // Counters
    integer in_cnt;
    integer out_cnt [0:3];
    integer err_cnt [0:3];
    integer i;
    integer total_errors;
    
    // REMOVED: skip_lat array

    // =========================================================================
    // 3. DUT INSTANTIATION
    // =========================================================================
    layer_2 dut (
        .clk(clk),
        .reset(reset),
        .pixel_in(pixel_in),   
        .valid_in(valid_in),
        .weight_b2(weight_b2),
        .bias_b2(bias_b2),
        .pixel_out(pixel_out),
        .ready(ready),
        .valid_out(valid_out)
    );

    // Clock
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // REMOVED: Valid Signal Delay Pipeline Block
    // REMOVED: Valid Signal Mux Block

    // =========================================================================
    // 5. STIMULUS
    // =========================================================================
    initial begin
        // --- A. LOAD DATA ---
        $display("[%0t] Loading Files...", $time);
        $readmemh({PATH, "golden_b1.hex"}, input_mem); 
        $readmemh({PATH, "golden_b2.hex"}, gold_mem); 
        $readmemh({PATH, "conv_b2_w.hex"}, weight_b2);
        $readmemh({PATH, "conv_b2_b.hex"}, bias_b2);

        // --- B. INITIALIZE ---
        reset = 1;
        valid_in = 0;
        in_cnt = 0;
        pixel_in[0] = 0; pixel_in[1] = 0; pixel_in[2] = 0; pixel_in[3] = 0;
        
        for(i=0; i<4; i++) begin
            out_cnt[i] = 0;
            err_cnt[i] = 0;
        end

        #100; reset = 0; #20;
        
        $display("[%0t] Starting Layer 2 Simulation...", $time);

        // --- C. FEED DATA ---
        while (in_cnt < IMG_IN_SIZE) begin
            @(negedge clk);
            if (ready) begin
                valid_in = 1;
                pixel_in[0] = input_mem[in_cnt];                      
                pixel_in[1] = input_mem[in_cnt + IMG_IN_SIZE];        
                pixel_in[2] = input_mem[in_cnt + 2*IMG_IN_SIZE];      
                pixel_in[3] = input_mem[in_cnt + 3*IMG_IN_SIZE];      
                in_cnt++;
            end else begin
                valid_in = 0;
            end
        end
        
        @(negedge clk);
        valid_in = 0;

        // --- D. WAIT FOR OUTPUTS ---
        $display("[%0t] Inputs finished. Waiting for %0d outputs...", $time, IMG_OUT_SIZE);
        
        fork
            begin : wait_done
                wait(out_cnt[0] == IMG_OUT_SIZE && out_cnt[1] == IMG_OUT_SIZE && 
                     out_cnt[2] == IMG_OUT_SIZE && out_cnt[3] == IMG_OUT_SIZE);
            end
            begin : timeout
                #100000;
                $display("\n[ERROR] Simulation Timed Out!");
                $display("Status: Ch0:%d Ch1:%d Ch2:%d Ch3:%d", out_cnt[0], out_cnt[1], out_cnt[2], out_cnt[3]);
                $finish;
            end
        join_any
        disable timeout;

        // --- E. FINAL REPORT ---
        $display("\n=======================================");
        $display("       RESULTS (ZERO DELAY)            ");
        $display("=======================================");
        total_errors = 0;
        for(i=0; i<4; i++) begin
             total_errors += err_cnt[i];
             $display("CH%0d: Processed %0d | Errors: %0d", i, out_cnt[i], err_cnt[i]);
        end

        if(total_errors == 0) 
            $display("\n[SUCCESS] Layer 2 Matches Golden!\n");
        else 
            $display("\n[FAILURE] Found %0d mismatches.\n", total_errors);
        
        $finish;
    end

    // =========================================================================
    // 6. CHECKER LOGIC (Direct Check)
    // =========================================================================
    genvar ch;
    generate
        for(ch=0; ch<4; ch++) begin : checkers
            always @(posedge clk) begin
                // Directly check valid_out from DUT
                if (valid_out[ch]) begin
                    
                    if (out_cnt[ch] < IMG_OUT_SIZE) begin
                        logic signed [15:0] hw;
                        logic signed [15:0] gold;
                        
                        hw = pixel_out[ch];
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