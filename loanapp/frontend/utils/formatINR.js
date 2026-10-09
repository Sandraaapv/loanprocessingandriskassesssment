/**
 * Lendr Financial Utilities: Indian Rupee (INR) Formatting
 * Single source of truth for all currency representations.
 * Strict adherence: No "$", no "k", no "M" suffixes.
 */

const inrFormatter = new Intl.NumberFormat("en-IN", {
  style: "currency",
  currency: "INR",
  maximumFractionDigits: 0
});

/**
 * Format plain number into full Indian Rupee representation (e.g. ₹1,00,000, ₹34,700).
 * @param {number} amount
 * @returns {string}
 */
export function formatINR(amount) {
  if (amount === null || amount === undefined || isNaN(amount)) return "₹0";
  return inrFormatter.format(Math.round(Number(amount)));
}

/**
 * Format plain number into Indian compact representation using Lakhs and Crores.
 * Examples:
 *   6000000  -> "₹60 Lakh"
 *   2400000  -> "₹24 Lakh"
 *   25000000 -> "₹2.5 Cr"
 *   4000000  -> "₹40 Lakh"
 *   800000   -> "₹8 Lakh"
 *   8400000  -> "₹84 Lakh"
 * @param {number} amount
 * @returns {string}
 */
export function formatINRCompact(amount) {
  if (amount === null || amount === undefined || isNaN(amount)) return "₹0";
  const num = Math.round(Number(amount));
  const abs = Math.abs(num);
  const sign = num < 0 ? "-" : "";

  if (abs >= 10000000) {
    const cr = (abs / 10000000).toFixed(2).replace(/\.?0+$/, "");
    return `${sign}₹${cr} Cr`;
  }
  if (abs >= 100000) {
    const lakh = (abs / 100000).toFixed(2).replace(/\.?0+$/, "");
    return `${sign}₹${lakh} Lakh`;
  }
  return formatINR(num);
}

export function formatLoanAmountThousands(amountInThousands) {
  if (amountInThousands === null || amountInThousands === undefined || isNaN(amountInThousands)) return "₹0";
  return formatINR(Number(amountInThousands) * 1000);
}

// Global window registration for standalone scripts
if (typeof window !== "undefined") {
  window.formatINR = formatINR;
  window.formatINRCompact = formatINRCompact;
  window.formatLoanAmountThousands = formatLoanAmountThousands;
}

