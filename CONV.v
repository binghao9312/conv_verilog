`timescale 1ns/10ps

module  CONV(
    input               clk,
    input               reset,
    output reg          busy,    
    input               ready,    
            
    output reg [11:0]   iaddr,
    input  signed [19:0] idata,    
    
    output reg          cwr,
    output reg [11:0]   caddr_wr,
    output reg [19:0]   cdata_wr,
    
    output reg          crd,
    output reg [11:0]   caddr_rd,
    input  signed [19:0] cdata_rd,
    
    output reg [2:0]    csel
);

// ==========================================
// Kernel 0 參數與 Bias (20-bit 有號數 Q4.16)
// ==========================================
localparam signed [19:0] k0 = 20'h0a89e; 
localparam signed [19:0] k1 = 20'h092d5;
localparam signed [19:0] k2 = 20'h06d43;
localparam signed [19:0] k3 = 20'h01004;
localparam signed [19:0] k4 = 20'hf8f71;
localparam signed [19:0] k5 = 20'hf6e54; 
localparam signed [19:0] k6 = 20'hfa6d7;
localparam signed [19:0] k7 = 20'hfc834;
localparam signed [19:0] k8 = 20'hfac19;

// Bias 值：01310 (16進制) [cite: 126]
// 擴充至 40 bits 以配合乘加器 (Q8.32) 的運算位元寬度
localparam signed [39:0] BIAS_40 = {4'h0, 20'h01310, 16'h0000}; 

// ==========================================
// FSM State Definition
// ==========================================
localparam  IDLE        = 3'd0,
            L0_FETCH    = 3'd1,
            L0_MAC_1    = 3'd2,
            L0_MAC_2    = 3'd3,
            L0_WRITE    = 3'd4,
            L1_FETCH    = 3'd5,
            L1_WRITE    = 3'd6,
            DONE        = 3'd7;

reg [2:0] CS, NS;

// ==========================================
// Registers & Counters
// ==========================================
reg [5:0] curr_x, curr_y;   // L0 座標 0~63
reg [4:0] l1_x, l1_y;       // L1 座標 0~31
reg [3:0] fetch_cnt;

reg signed [19:0] LB [0:8]; // 儲存 3x3 window 內的 9 個像素
reg is_pad_reg;             // 延遲一拍的 Padding 標記

// Pipeline 加法樹暫存器
reg signed [39:0] adder_seq_0;
reg signed [39:0] adder_seq_1;
reg signed [39:0] s8_seq;
reg signed [39:0] adder_seq_final;

reg signed [19:0] l1_max;   // 紀錄 Max-pooling 的最大值 [cite: 297]

// ==========================================
// Data Path (Layer 0 MAC & Pipeline)
// ==========================================
// T0 階段 (組合邏輯乘法與第一級加法)
wire signed [39:0] s0 = LB[0] * k0;
wire signed [39:0] s1 = LB[1] * k1;
wire signed [39:0] s2 = LB[2] * k2;
wire signed [39:0] s3 = LB[3] * k3;
wire signed [39:0] s4 = LB[4] * k4;
wire signed [39:0] s5 = LB[5] * k5;
wire signed [39:0] s6 = LB[6] * k6;
wire signed [39:0] s7 = LB[7] * k7;
wire signed [39:0] s8 = LB[8] * k8;

wire signed [39:0] adder0 = s0 + s1;
wire signed [39:0] adder1 = s2 + s3;
wire signed [39:0] adder2 = s4 + s5;
wire signed [39:0] adder3 = s6 + s7;

// 最終結果加上 Bias 並於第 17 位做四捨五入 [cite: 260]
wire signed [39:0] sum_with_bias = adder_seq_final + BIAS_40;
wire signed [39:0] rounded_sum   = sum_with_bias + 40'h0000_0000_8000; 
wire signed [19:0] l0_out_pre    = rounded_sum[35:16];

// ReLU：若為負數 (MSB為1) 則輸出 0 [cite: 257]
wire signed [19:0] l0_out_relu   = (rounded_sum[39]) ? 20'd0 : l0_out_pre;

// ==========================================
// Kernel Coordinates & Zero-padding 判斷
// ==========================================
reg [1:0] kx, ky;
always @(*) begin
    case(fetch_cnt)
        0: begin kx = 0; ky = 0; end
        1: begin kx = 1; ky = 0; end
        2: begin kx = 2; ky = 0; end
        3: begin kx = 0; ky = 1; end
        4: begin kx = 1; ky = 1; end
        5: begin kx = 2; ky = 1; end
        6: begin kx = 0; ky = 2; end
        7: begin kx = 1; ky = 2; end
        8: begin kx = 2; ky = 2; end
        default: begin kx = 0; ky = 0; end
    endcase
end

// 計算記憶體讀取座標，超出 0~63 範圍則觸發 Padding 
wire signed [7:0] tx = $signed({2'b00, curr_x}) + $signed({6'b00, kx}) - 8'sd1;
wire signed [7:0] ty = $signed({2'b00, curr_y}) + $signed({6'b00, ky}) - 8'sd1;
wire out_of_bounds = (tx < 0) || (tx > 63) || (ty < 0) || (ty > 63);

// ==========================================
// FSM Next State Logic (Ternary & Default)
// ==========================================
always @(posedge clk or posedge reset) begin
    if (reset) CS <= IDLE;
    else       CS <= NS;
end

always @(*) begin
    case (CS)
        IDLE:       
            NS = ready ? L0_FETCH : IDLE;
            
        L0_FETCH:   
            NS = (fetch_cnt == 4'd9) ? L0_MAC_1 : L0_FETCH;
            
        L0_MAC_1:   
            NS = L0_MAC_2;
            
        L0_MAC_2:   
            NS = L0_WRITE;
            
        L0_WRITE:   
            NS = (curr_x == 6'd63 && curr_y == 6'd63) ? L1_FETCH : L0_FETCH;
            
        L1_FETCH:   
            NS = (fetch_cnt == 4'd4) ? L1_WRITE : L1_FETCH;
            
        L1_WRITE:   
            NS = (l1_x == 5'd31 && l1_y == 5'd31) ? DONE : L1_FETCH;
            
        DONE:       
            NS = IDLE;
            
        default:    
            NS = IDLE;
    endcase
end

// ==========================================
// Sequential Logic
// ==========================================
integer i;
always @(posedge clk or posedge reset) begin
    if (reset) begin
        busy <= 1'b0;
        curr_x <= 6'd0;
        curr_y <= 6'd0;
        l1_x <= 5'd0;
        l1_y <= 5'd0;
        fetch_cnt <= 4'd0;
        is_pad_reg <= 1'b0;
        for (i=0; i<9; i=i+1) LB[i] <= 20'd0;
        
        adder_seq_0 <= 40'd0;
        adder_seq_1 <= 40'd0;
        s8_seq <= 40'd0;
        adder_seq_final <= 40'd0;
        l1_max <= 20'd0;

        cwr <= 1'b0;
        crd <= 1'b0;
        caddr_wr <= 12'd0;
        cdata_wr <= 20'd0;
        caddr_rd <= 12'd0;
        csel <= 3'd0;
        iaddr <= 12'd0;
    end else begin
        case (CS)
            IDLE: begin
                if (ready) begin
                    busy <= 1'b1; // 通知 Testfixture 開始動作 [cite: 50]
                    curr_x <= 6'd0;
                    curr_y <= 6'd0;
                    l1_x <= 5'd0;
                    l1_y <= 5'd0;
                    fetch_cnt <= 4'd0;
                end
            end

            L0_FETCH: begin
                // 發送讀取記憶體位址 (SRAM Read Takes 1 Cycle)
                if (fetch_cnt < 9) begin
                    iaddr <= {ty[5:0], tx[5:0]}; // 等同於 ty*64 + tx
                    is_pad_reg <= out_of_bounds;
                end
                // 晚一個 Cycle 將記憶體回傳資料存入 Line Buffer
                if (fetch_cnt > 0 && fetch_cnt <= 9) begin
                    LB[fetch_cnt - 1] <= is_pad_reg ? 20'd0 : idata;
                end
                
                if (fetch_cnt == 9) fetch_cnt <= 4'd0;
                else fetch_cnt <= fetch_cnt + 1;
                
                cwr <= 1'b0; 
            end

            L0_MAC_1: begin
                adder_seq_0 <= adder0 + adder1;
                adder_seq_1 <= adder2 + adder3;
                s8_seq      <= s8;
            end

            L0_MAC_2: begin
                adder_seq_final <= adder_seq_0 + adder_seq_1 + s8_seq;
            end

            L0_WRITE: begin
                cwr <= 1'b1;
                csel <= 3'b001; // 啟動 L0_MEM0 記憶體 [cite: 259]
                caddr_wr <= {curr_y, curr_x};
                cdata_wr <= l0_out_relu;

                // L0 座標推進 [cite: 121]
                if (curr_x == 6'd63) begin
                    curr_x <= 6'd0;
                    if (curr_y == 6'd63) curr_y <= 6'd0;
                    else curr_y <= curr_y + 1;
                end else begin
                    curr_x <= curr_x + 1;
                end
            end

            L1_FETCH: begin
                cwr <= 1'b0;
                // 發送 L0_MEM0 讀取位址 (步幅 Stride = 2) [cite: 297]
                if (fetch_cnt < 4) begin
                    crd <= 1'b1;
                    csel <= 3'b001; 
                    case(fetch_cnt)
                        0: caddr_rd <= {l1_y, 1'b0, l1_x, 1'b0}; // y*2,   x*2
                        1: caddr_rd <= {l1_y, 1'b0, l1_x, 1'b1}; // y*2,   x*2+1
                        2: caddr_rd <= {l1_y, 1'b1, l1_x, 1'b0}; // y*2+1, x*2
                        3: caddr_rd <= {l1_y, 1'b1, l1_x, 1'b1}; // y*2+1, x*2+1
                    endcase
                end else begin
                    crd <= 1'b0;
                end

                // 晚一個 Cycle 接收資料並找尋 2x2 Window 內最大值
                if (fetch_cnt == 1) begin
                    l1_max <= cdata_rd; 
                end 
                else if (fetch_cnt > 1 && fetch_cnt <= 4) begin
                    if (cdata_rd > l1_max) l1_max <= cdata_rd;
                end

                if (fetch_cnt == 4) fetch_cnt <= 4'd0;
                else fetch_cnt <= fetch_cnt + 1;
            end

            L1_WRITE: begin
                cwr <= 1'b1;
                csel <= 3'b011; // 啟動 L1_MEM0 記憶體 [cite: 347]
                caddr_wr <= {2'b00, l1_y, l1_x}; // {2位補零, Y座標, X座標}
                cdata_wr <= l1_max;

                // L1 座標推進 [cite: 297]
                if (l1_x == 5'd31) begin
                    l1_x <= 5'd0;
                    if (l1_y == 5'd31) l1_y <= 5'd0;
                    else l1_y <= l1_y + 1;
                end else begin
                    l1_x <= l1_x + 1;
                end
            end

            DONE: begin
                busy <= 1'b0; // 運算結束通知 Testfixture [cite: 50]
                cwr <= 1'b0;
            end
        endcase
    end
end

endmodule