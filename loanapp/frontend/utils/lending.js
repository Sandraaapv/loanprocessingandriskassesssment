/**
 * Lendr Lending & Eligibility Engine
 * Single source of truth for all loan calculations, FOIR constraints, and underwriting assessments.
 */

import { formatINR, formatINRCompact } from "./formatINR.js";

export const MAX_FOIR = 45; // Regulatory / prudent retail banking limit: 45% of income
export const ADVISORY_LOAN_THRESHOLD = 2500000; // Rs 25,00,000 threshold

/**
 * Standard reducing-balance EMI formula
 * E = P * r * (1+r)^n / ((1+r)^n - 1)
 * @param {number} principal
 * @param {number} annualRatePct
 * @param {number} tenureYears
 * @returns {number} monthly EMI in Rupees
 */
export function emi(principal, annualRatePct, tenureYears) {
  const p = Number(principal);
  const rAnnual = Number(annualRatePct);
  const nYears = Number(tenureYears);
  if (p <= 0 || nYears <= 0) return 0;
  if (rAnnual <= 0) return Math.round(p / (nYears * 12));

  const r = rAnnual / 12 / 100;
  const n = nYears * 12;
  const factor = Math.pow(1 + r, n);
  const monthlyEmi = (p * r * factor) / (factor - 1);
  return Math.round(monthlyEmi);
}

/**
 * Fixed Obligation to Income Ratio (FOIR) percentage
 * @param {number} totalMonthlyEmi
 * @param {number} monthlyIncome
 * @returns {number} FOIR as a percentage (0 - 100)
 */
export function foir(totalMonthlyEmi, monthlyIncome) {
  const inc = Number(monthlyIncome);
  if (inc <= 0) return 100;
  const ratio = (Number(totalMonthlyEmi) / inc) * 100;
  return Math.round(ratio * 10) / 10;
}

/**
 * The largest principal where (existingEmi + newEmi) / monthlyIncome <= MAX_FOIR
 * @param {number} monthlyIncome
 * @param {number} existingEmi
 * @param {number} annualRatePct
 * @param {number} tenureYears
 * @returns {number} Maximum eligible loan principal in Rupees
 */
export function maxEligibleLoan(monthlyIncome, existingEmi = 0, annualRatePct = 8.5, tenureYears = 20) {
  const inc = Number(monthlyIncome);
  const exist = Number(existingEmi);
  const maxAllowedEmi = inc * (MAX_FOIR / 100);
  const availableEmi = Math.max(0, maxAllowedEmi - exist);

  if (availableEmi <= 0 || tenureYears <= 0) return 0;

  const rAnnual = Number(annualRatePct);
  if (rAnnual <= 0) {
    return Math.round(availableEmi * tenureYears * 12);
  }

  const r = rAnnual / 12 / 100;
  const n = Number(tenureYears) * 12;
  const factor = Math.pow(1 + r, n);

  // P = E * (factor - 1) / (r * factor)
  const principal = (availableEmi * (factor - 1)) / (r * factor);
  return Math.round(principal);
}

/**
 * Assess a loan application against applicant financials
 * @param {Object} application - { amount, annualRatePct, tenureYears, purpose }
 * @param {Object} applicant - { monthlyIncome, existingEmi, name }
 * @returns {Object} { decision, foirAfter, headroomEmi, maxEligibleAmount, newEmi, reason }
 */
export function assess(application, applicant) {
  const requestedPrincipal = Number(application.amount || 0);
  const rate = Number(application.annualRatePct || 8.5);
  const tenure = Number(application.tenureYears || 20);
  const income = Number(applicant.monthlyIncome || 100000);
  const currentEmi = Number(applicant.existingEmi || 0);

  const newEmi = emi(requestedPrincipal, rate, tenure);
  const totalEmiAfter = currentEmi + newEmi;
  const foirAfter = foir(totalEmiAfter, income);
  const maxAllowedEmi = income * (MAX_FOIR / 100);
  const headroomEmi = Math.max(0, maxAllowedEmi - currentEmi);
  const maxEligibleAmount = maxEligibleLoan(income, currentEmi, rate, tenure);

  let decision = "approved";
  let reason = "";

  if (foirAfter <= MAX_FOIR) {
    decision = "approved";
    reason = `Approved. EMI load of ${Math.round(foirAfter)}% remains safely within the ${MAX_FOIR}% limit.`;
  } else if (maxEligibleAmount >= 100000) {
    decision = "approved_reduced";
    reason = `Requested amount exceeds limit. Maximum eligible loan is ${formatINRCompact(maxEligibleAmount)} to maintain FOIR within ${MAX_FOIR}%.`;
  } else {
    decision = "rejected";
    reason = `EMI load would reach ${Math.round(foirAfter)}% of income, above the ${MAX_FOIR}% limit.`;
  }

  return {
    decision,
    foirAfter,
    headroomEmi,
    maxEligibleAmount,
    newEmi,
    reason
  };
}

// Global window registration for standalone scripts
if (typeof window !== "undefined") {
  window.MAX_FOIR = MAX_FOIR;
  window.ADVISORY_LOAN_THRESHOLD = ADVISORY_LOAN_THRESHOLD;
  window.lendingEngine = {
    emi,
    foir,
    MAX_FOIR,
    ADVISORY_LOAN_THRESHOLD,
    maxEligibleLoan,
    assess
  };
}
