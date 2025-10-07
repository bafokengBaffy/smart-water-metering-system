const express = require("express");
const app = express();
app.use(express.json());

// Middleware for logging requests
app.use((req, res, next) => {
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.url}`);
  next();
});

// Mock payment processing
app.post("/process-payment", (req, res) => {
  const { phone, amount, provider } = req.body;

  // Input validation
  if (!phone || !amount || !provider) {
    return res.status(400).json({ success: false, message: "Invalid request: Missing required fields" });
  }

  if (isNaN(amount) || amount <= 0) {
    return res.status(400).json({ success: false, message: "Invalid amount: Must be a positive number" });
  }

  if (!["mpesa", "ecocash"].includes(provider.toLowerCase())) {
    return res.status(400).json({ success: false, message: "Invalid provider: Supported providers are M-Pesa and EcoCash" });
  }

  // Simulate payment processing
  console.log(`Processing payment of ${amount} to ${phone} via ${provider}...`);
  setTimeout(() => {
    console.log(`Payment of ${amount} to ${phone} via ${provider} processed successfully!`);
    res.json({
      success: true,
      message: `Payment of ${amount} to ${phone} via ${provider} processed!`,
    });
  }, 2000); // Simulate a 2-second delay
});

// Callback endpoint for M-Pesa
app.post("/mpesa-callback", (req, res) => {
  const callbackData = req.body;

  if (!callbackData) {
    return res.status(400).json({ success: false, message: "Invalid callback data" });
  }

  console.log("M-Pesa callback received:", callbackData);

  // Check if the payment was successful
  if (callbackData.Body?.stkCallback?.ResultCode === 0) {
    console.log("Payment successful!");
    console.log("Receipt Number:", callbackData.Body.stkCallback.CallbackMetadata.Item[1].Value);
    console.log("Amount:", callbackData.Body.stkCallback.CallbackMetadata.Item[0].Value);
  } else {
    console.log("Payment failed:", callbackData.Body?.stkCallback?.ResultDesc);
  }

  // Respond to M-Pesa
  res.status(200).json({ success: true, message: "Callback received" });
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(`[${new Date().toISOString()}] Error:`, err.message);
  res.status(500).json({ success: false, message: "Internal server error" });
});

// Start the server
const PORT = 3000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});