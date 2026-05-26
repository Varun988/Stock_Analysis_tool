# Manual API Test Flow

## Purpose

This document provides a repeatable manual API test flow for the Stock Analysis Tool backend.

Use this checklist after backend changes to verify that the core in-memory API flow is working correctly.

## Current Backend Flow

```text
Profile
→ Instrument
→ Market Data
→ Metrics
→ Risk Engine
→ Portfolio Holding
→ Recommendation
→ Explanation