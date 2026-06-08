const { NseIndia } = require("stock-nse-india");

const nseIndia = new NseIndia();

const holdings = [
  {
    statementName: "HDFCAMC - HDFCNIFTY",
    isin: "INF179KC1965",
    candidateSymbols: ["HDFCNIFTY"],
  },
  {
    statementName: "ICICI PRUDENTIAL NV20 ETF",
    isin: "INF109KC11V0",
    candidateSymbols: ["NV20IETF"],
  },
  {
    statementName: "NIP IND ETF NIFTY BEES",
    isin: "INF204KB14I2",
    candidateSymbols: ["NIFTYBEES"],
  },
  {
    statementName: "SBI-ETF NIFTY 50",
    isin: "INF200KA1FS1",
    candidateSymbols: ["SETFNIF50"],
  },
];

async function testSymbol(symbol) {
  console.log("\n----------------------------------------");
  console.log("Testing NSE symbol:", symbol);
  console.log("----------------------------------------");

  try {
    console.log("\nFetching equity details...");
    const details = await nseIndia.getEquityDetails(symbol);
    console.dir(details, { depth: 6 });
  } catch (error) {
    console.error("Equity details failed:", error.message);
  }

  try {
    console.log("\nFetching historical data...");
    const range = {
      start: new Date("2024-01-01"),
      end: new Date(),
    };

    const history = await nseIndia.getEquityHistoricalData(symbol, range);

    console.log("Historical response type:", Array.isArray(history) ? "array" : typeof history);
    console.log("Historical rows:", Array.isArray(history) ? history.length : "not-array");

    if (Array.isArray(history) && history.length > 0) {
      console.log("First historical row:");
      console.dir(history[0], { depth: 4 });

      console.log("Latest historical row:");
      console.dir(history[history.length - 1], { depth: 4 });
    }
  } catch (error) {
    console.error("Historical data failed:", error.message);
  }
}

async function main() {
  console.log("Fetching all NSE stock symbols...");
  const allSymbols = await nseIndia.getAllStockSymbols();

  console.log("Total NSE symbols:", allSymbols.length);

  for (const holding of holdings) {
    console.log("\n========================================");
    console.log("Statement name:", holding.statementName);
    console.log("ISIN:", holding.isin);
    console.log("Candidate symbols:", holding.candidateSymbols.join(", "));
    console.log("========================================");

    for (const symbol of holding.candidateSymbols) {
      console.log("Exists in getAllStockSymbols:", allSymbols.includes(symbol));
      await testSymbol(symbol);
    }
  }
}

main().catch((error) => {
  console.error("Fatal error:", error);
});