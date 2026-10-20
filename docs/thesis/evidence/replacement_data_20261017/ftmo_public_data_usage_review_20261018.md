# FTMO public data-use and publication review

**Reviewed:** 2026-10-18 15:55 Europe/Helsinki  
**Purpose:** Prepare a bounded source and usage disclosure for the thesis before P11. This is a review of public FTMO materials, not a legal opinion or confirmation of account-specific rights.

## Existing source record

The user attests that the US100.cash and US500.cash CSV exports came from an FTMO Demo MetaTrader 5 server. The exact server and liquidity provider are unknown. No account-specific export, data license, or publication permission record was supplied. The user considers the files suitable for model training. See [`data_source_user_attestation_20261018.json`](data_source_user_attestation_20261018.json).

## Public materials reviewed

- [FTMO Terms and Conditions](https://ftmo.com/en/terms-and-conditions/), accessed 2026-10-18. The current public page describes FTMO as providing simulated trading, says its platform and data-feed technical solution is powered by liquidity providers, and states that FTMO accounts are demo accounts with fictitious funds. The page does not identify the provider for these exports or set out a CSV export or thesis-publication license.
- [How FTMO works](https://ftmo.com/en/blog/how-ftmo-works/), accessed 2026-10-18; published approximately three years before this review. It describes demo accounts as providing information on real-market development and says liquidity providers supply instrument price data in the simulated environment. This is general platform context, not proof of the specific source or terms for these files.
- [FTMO FAQ on the legal relationship after signing FTMO Account Terms & Conditions](https://ftmo.com/en/faq/what-is-the-legal-relationship-between-an-ftmo-trader-and-ftmo-after-signing-the-ftmo-account-terms-conditions/), accessed 2026-10-18. FTMO describes the account terms as binding and says a sample contract may be requested. It does not establish what agreement applies to this Demo account or grant rights for these exports.

## Finding and boundary

The public materials are consistent with the user's FTMO Demo source declaration and the use of externally supplied price data in FTMO's simulated platform. They do not independently authenticate the CSVs, identify their liquidity provider, establish the account-specific export path or applicable historical terms, or grant permission to reproduce raw bars/ticks or publish results derived from them. The account-specific publication and data-use status therefore remains unverified. This review did not locate an explicit public authorization for the intended thesis use; that absence is not proof that no permission exists.

## Draft thesis disclosure

> The US100.cash and US500.cash price and tick files used in this study were exported from an FTMO Demo MetaTrader 5 server, as attested by the researcher. FTMO's public materials describe its platform as simulated and its data feed as powered by liquidity providers; the specific upstream provider and exact Demo server configuration for these historical exports were not established. No account-specific export or publication license record was available for this study. The files were used for model training and are not included in the thesis submission package. Results for 1 January–16 October 2026 are exploratory/report-only because that interval had previously been inspected. These source and licensing limits constrain independent verification and data redistribution.

This wording reports the evidence boundary; it does not assert that publication or redistribution is contractually permitted. Before P11, reconcile it with the applicable account terms and supervisor or university requirements. If permission remains undocumented, retain the disclosure and do not distribute the raw exports with the submission.
