`timescale 1ns / 1ps

module tb_layer_all;

    // =========================================================================
    // 1. CONFIGURATION
    // =========================================================================
    parameter string PATH = "D:/Senior/CNN_test/data/";
    
    // Input Image Size (28x28 = 784)
    parameter int IMG_IN_SIZE = 784; 
    
    // =========================================================================
    // 2. SIGNALS
    // =========================================================================
    logic clk, reset;
    
    // --- DUT Inputs ---
    logic signed [15:0] pixel_in;
    logic valid_in;
    
    // --- Weights & Biases (Layers 1-4) ---
    logic signed [15:0] weight_b1 [0:3][0:8];
    logic signed [31:0] bias_b1 [0:3];
    
    logic signed [15:0] weight_b2 [0:3][0:3][0:8];
    logic signed [31:0] bias_b2 [0:3];
    
    logic signed [15:0] weight_b3 [0:3][0:3][0:8];
    logic signed [31:0] bias_b3 [0:3];
    
    logic signed [15:0] weight_b4 [0:3][0:3][0:8];
    logic signed [31:0] bias_b4 [0:3];
    
    // --- Weights & Biases (Classifier) ---
    logic signed [15:0] weight_class [0:9][0:3][0:48];
    logic signed [31:0] bias_class [0:9];

    // --- DUT Outputs ---
    logic [3:0] result;
    logic ready;
    logic done;

    // --- Memories & Variables ---
    logic signed [15:0] img_mem [0:783]; // Input Image Memory
    
    integer in_cnt;
    integer timeout;

    // =========================================================================
    // 3. DUT INSTANTIATION
    // =========================================================================
    cnn_top dut (
        .clk(clk),
        .reset(reset),
        .pixel_in(pixel_in),
        .valid_in(valid_in),
        
        // Connect CNN Layers 1-4
        .weight_b1(weight_b1), .bias_b1(bias_b1),
        .weight_b2(weight_b2), .bias_b2(bias_b2),
        .weight_b3(weight_b3), .bias_b3(bias_b3),
        .weight_b4(weight_b4), .bias_b4(bias_b4),
        
        // Connect Classifier
        .weight_class(weight_class), .bias_class(bias_class),
        
        // Outputs
        .result(result),
        .ready(ready),
        .done(done)
    );

    // Clock Generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // =========================================================================
    // 4. STIMULUS & CONTROL
    // =========================================================================
    initial begin
        // --- A. LOAD FILES ---
        $display("[%0t] Loading Files from %s...", $time, PATH);
        
        // 1. Load Input Image
        $readmemh({PATH, "test_image_2.hex"}, img_mem);
        
        // 2. Load Layer 1-4 Parameters
        $readmemh({PATH, "conv_b1_w.hex"}, weight_b1);
        $readmemh({PATH, "conv_b1_b.hex"}, bias_b1);
        
        $readmemh({PATH, "conv_b2_w.hex"}, weight_b2);
        $readmemh({PATH, "conv_b2_b.hex"}, bias_b2);
        
        $readmemh({PATH, "conv_b3_w.hex"}, weight_b3);
        $readmemh({PATH, "conv_b3_b.hex"}, bias_b3);
        
        $readmemh({PATH, "conv_b4_w.hex"}, weight_b4);
        $readmemh({PATH, "conv_b4_b.hex"}, bias_b4);
        
        // 3. Load Classifier Parameters
        $readmemh({PATH, "classifier_w.hex"}, weight_class);
        $readmemh({PATH, "classifier_b.hex"}, bias_class);

        // --- B. INITIALIZE ---
        reset = 1;
        valid_in = 0;
        pixel_in = 0;
        in_cnt = 0;
        
        #100; // Hold reset
        reset = 0; 
        #20;

        $display("[%0t] Reset released. Starting Full CNN Inference...", $time);

        // --- C. FEED DATA ---
        // Feed pixels one by one when the DUT is ready
        while (in_cnt < IMG_IN_SIZE) begin
            @(negedge clk);
            if (ready) begin
                valid_in = 1;
                pixel_in = img_mem[in_cnt];
                in_cnt = in_cnt + 1;
            end else begin
                // If DUT is busy (processing layers), hold valid low
                valid_in = 0;
            end
        end
        
        // Stop sending data
        @(negedge clk);
        valid_in = 0;
        pixel_in = 0;
        $display("[%0t] All %0d pixels sent. Waiting for classification...", $time, IMG_IN_SIZE);

        // --- D. WAIT FOR RESULT ---
        timeout = 0;
        while (!done && timeout < 1000000) begin
            @(posedge clk);
            timeout = timeout + 1;
        end

        if (timeout >= 1000000) begin
            $display("\n[ERROR] Simulation Timed Out! 'done' signal never went high.");
            $finish;
        end

        // Wait one extra cycle to ensure result is stable
        @(posedge clk);

        // --- E. FINAL REPORT ---
        $display("\n================================================");
        $display("          FULL CNN CLASSIFICATION RESULT        ");
        $display("================================================");
        $display(" Image Processed: test_image_0.hex");
        $display(" Time Taken     : %0t", $time);
        $display(" ----------------------------------------------");
        $display(" PREDICTED DIGIT : %0d", result);
        $display(" ================================================\n");

        $finish;
    end

endmodule