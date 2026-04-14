# Payment API

This document explains the payment API surface area and the main concepts used by the service. It is intended to provide enough context for implementers and reviewers to work safely with the integration points described below.

## Authorization

The authorization flow accepts a request, validates merchant settings, checks risk controls, and places a temporary hold on available funds. The response contains an approval or decline result and references that downstream systems use for capture and reversal.

## Capture

The capture flow converts a prior authorization hold into a posted financial record. Partial capture is supported when a merchant submits a smaller amount than originally approved and the remaining hold is released automatically.
