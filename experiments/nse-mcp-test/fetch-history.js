const { NseIndia } = require("stock-nse-india");

const nseIndia = new NseIndia();

function normalizeHistoryResponse(history) {
  if (!history) {
    return [];
  }

  if (Array.isArray(history)) {
    // Case 1: direct array of NSE rows
    if (history.length > 0 && history[0]?.mtimestamp) {
      return history;
    }

    // Case 2: array of chunks: [{ data: [...], meta: {...} }]
    return history.flatMap((item) => {
      if (Array.isArray(item?.data)) {
        return item.data;
      }
      return [];
    });
  }

  // Case 3: object: { data: [...], meta: {...} }
  if (Array.isArray(history.data)) {
    return history.data;
  }

  return [];
}

function toIsoDate(nseDate) {
  // NSE date example: "05-Jun-2026"
  const parsed = new Date(nseDate);
  if (Number.isNaN(parsed.getTime())) {
    return null;
  }
  return parsed.toISOString().slice(0, 10);
}

function normalizeRow(row) {
  return {
    data_date: toIsoDate(row.mtimestamp),
    open_price: row.chOpeningPrice ?? null,
    high_price: row.chTradeHighPrice ?? null,
    low_price: row.chTradeLowPrice ?? null,
    close_price: row.chClosingPrice ?? null,
    volume: row.chTotTradedQty ?? null,
    symbol: row.chSymbol ?? null,
    source_payload: row,
  };
}

async function fetchHistory(symbol, startDate, endDate) {
  const range = {
    start: new Date(startDate),
    end: new Date(endDate),
  };

  const history = await nseIndia.getEquityHistoricalData(symbol, range);
  const rows = normalizeHistoryResponse(history)
    .map(normalizeRow)
    .filter((row) => row.data_date && row.close_price !== null);

  rows.sort((a, b) => a.data_date.localeCompare(b.data_date));

  return rows;
}

async function main() {
  const symbol = process.argv[2];
  const startDate = process.argv[3];
  const endDate = process.argv[4];

  if (!symbol || !startDate || !endDate) {
    console.error(
      JSON.stringify({
        success: false,
        error: "Usage: node fetch-history.js SYMBOL YYYY-MM-DD YYYY-MM-DD",
      })
    );
    process.exit(1);
  }

  try {
    const rows = await fetchHistory(symbol, startDate, endDate);

    console.log(
      JSON.stringify(
        {
          success: true,
          symbol,
          startDate,
          endDate,
          rowCount: rows.length,
          firstRow: rows[0] || null,
          latestRow: rows[rows.length - 1] || null,
          rows,
        },
        null,
        2
      )
    );
  } catch (error) {
    console.error(
      JSON.stringify(
        {
          success: false,
          symbol,
          error: error.message,
        },
        null,
        2
      )
    );
    process.exit(1);
  }
}

main();