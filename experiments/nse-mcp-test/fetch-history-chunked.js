const { NseIndia } = require("stock-nse-india");

const nseIndia = new NseIndia();

function normalizeHistoryResponse(history) {
  if (!history) return [];

  if (Array.isArray(history)) {
    if (history.length > 0 && history[0]?.mtimestamp) {
      return history;
    }

    return history.flatMap((item) => {
      if (Array.isArray(item?.data)) return item.data;
      return [];
    });
  }

  if (Array.isArray(history.data)) return history.data;

  return [];
}

function toIsoDate(nseDate) {
  const parsed = new Date(nseDate);
  if (Number.isNaN(parsed.getTime())) return null;
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

function addDays(date, days) {
  const next = new Date(date);
  next.setDate(next.getDate() + days);
  return next;
}

function minDate(a, b) {
  return a <= b ? a : b;
}

function toYyyyMmDd(date) {
  return date.toISOString().slice(0, 10);
}

async function fetchChunk(symbol, start, end) {
  const range = {
    start,
    end,
  };

  const history = await nseIndia.getEquityHistoricalData(symbol, range);

  return normalizeHistoryResponse(history)
    .map(normalizeRow)
    .filter((row) => row.data_date && row.close_price !== null);
}

async function fetchHistoryChunked(symbol, startDate, endDate, chunkDays = 30) {
  const start = new Date(startDate);
  const end = new Date(endDate);

  let cursor = start;
  const allRows = [];
  const errors = [];

  while (cursor <= end) {
    const chunkStart = new Date(cursor);
    const chunkEnd = minDate(addDays(chunkStart, chunkDays - 1), end);

    try {
      const rows = await fetchChunk(symbol, chunkStart, chunkEnd);
      allRows.push(...rows);
    } catch (error) {
      errors.push({
        symbol,
        startDate: toYyyyMmDd(chunkStart),
        endDate: toYyyyMmDd(chunkEnd),
        error: error.message,
      });
    }

    cursor = addDays(chunkEnd, 1);

    // Be polite to NSE endpoints.
    await new Promise((resolve) => setTimeout(resolve, 500));
  }

  const dedupedMap = new Map();

  for (const row of allRows) {
    dedupedMap.set(row.data_date, row);
  }

  const rows = Array.from(dedupedMap.values()).sort((a, b) =>
    a.data_date.localeCompare(b.data_date)
  );

  return {
    rows,
    errors,
  };
}

async function main() {
  const symbol = process.argv[2];
  const startDate = process.argv[3];
  const endDate = process.argv[4];

  if (!symbol || !startDate || !endDate) {
    console.error(
      JSON.stringify({
        success: false,
        error:
          "Usage: node fetch-history-chunked.js SYMBOL YYYY-MM-DD YYYY-MM-DD",
      })
    );
    process.exit(1);
  }

  const result = await fetchHistoryChunked(symbol, startDate, endDate, 30);

  console.log(
    JSON.stringify(
      {
        success: result.rows.length > 0,
        symbol,
        startDate,
        endDate,
        rowCount: result.rows.length,
        errorCount: result.errors.length,
        errors: result.errors,
        firstRow: result.rows[0] || null,
        latestRow: result.rows[result.rows.length - 1] || null,
        rows: result.rows,
      },
      null,
      2
    )
  );
}

main().catch((error) => {
  console.error(
    JSON.stringify(
      {
        success: false,
        error: error.message,
      },
      null,
      2
    )
  );
  process.exit(1);
});