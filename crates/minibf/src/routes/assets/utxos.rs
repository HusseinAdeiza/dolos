//! `/assets/{asset}/utxos`: the live UTxOs holding an asset.

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Json,
};
use blockfrost_openapi::models::asset_utxo_content_inner::AssetUtxoContentInner;
use dolos_cardano::indexes::CardanoStateIndexExt as _;
use dolos_cardano::model::AssetState;
use dolos_core::Domain;

use super::resolve_asset_state;
use crate::{
    error::Error,
    pagination::{Pagination, PaginationParameters},
    Facade,
};

/// `GET /assets/{subject}/utxos`
///
/// Live UTxOs that carry the asset, in chain order. `from` / `to` bound the
/// creation block height, inclusive; an optional `:index` narrows the edge
/// block by tx index, as on the other endpoints. A valid but unknown asset is a
/// 404; a known asset that no live UTxO holds any more (fully burned) is an
/// empty page. UTxOs whose creation block was pruned by `sync.max_history` are
/// not listed: the row requires block data the node no longer has.
///
/// `max_scan_items` caps the page depth, like on the other asset lists.
pub async fn by_subject_utxos<D>(
    Path(subject): Path<String>,
    Query(params): Query<PaginationParameters>,
    State(domain): State<Facade<D>>,
) -> Result<Json<Vec<AssetUtxoContentInner>>, Error>
where
    D: Domain + Clone + Send + Sync + 'static,
    Option<AssetState>: From<D::Entity>,
{
    let pagination = Pagination::try_from(params)?;
    pagination.enforce_max_scan_limit(domain.config.max_scan_items())?;
    let (asset, _) = resolve_asset_state(&domain, &subject)?;

    let refs = domain
        .state()
        .utxos_by_asset(&asset)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let utxos =
        crate::routes::utxos::load_utxo_models_in_height_range(&domain, refs, pagination).await?;

    Ok(Json(utxos))
}

#[cfg(test)]
mod tests {
    use super::super::testing::*;
    use super::*;
    use crate::test_support::{TestApp, TestFault};
    use dolos_testing::synthetic::SyntheticBlockConfig;
    use itertools::Itertools as _;

    fn parse_utxos(bytes: &[u8]) -> Vec<AssetUtxoContentInner> {
        serde_json::from_slice(bytes).expect("failed to parse asset utxos")
    }

    #[tokio::test]
    async fn assets_by_subject_utxos_happy_path() {
        let app = TestApp::new();
        let asset = app.vectors().asset_unit.as_str();
        let path = format!("/assets/{asset}/utxos");
        let (status, bytes) = app.get_bytes(&path).await;

        assert_eq!(
            status,
            StatusCode::OK,
            "unexpected status {status} with body: {}",
            String::from_utf8_lossy(&bytes)
        );
        let items = parse_utxos(&bytes);
        assert!(!items.is_empty());

        for item in items {
            // every UTxO must carry the asset and sit where the chain put it
            assert!(item.amount.iter().any(|x| x.unit == asset));
            assert!(item.amount.iter().any(|x| x.unit == "lovelace"));
            let (block_number, _) = app.vectors().tx_position(&item.tx_hash);
            assert_eq!(item.block_height as u64, block_number);
            assert!(!item.block.is_empty());
            assert!(item.block_time > 0);
            assert_eq!(item.inline_datum, None);
            assert_eq!(item.inline_datum_json, None);
        }
    }

    #[tokio::test]
    async fn assets_by_subject_utxos_paginated() {
        let app = TestApp::new();
        let asset = app.vectors().asset_unit.as_str();

        let (status_1, bytes_1) = app
            .get_bytes(&format!("/assets/{asset}/utxos?page=1&count=1"))
            .await;
        let (status_2, bytes_2) = app
            .get_bytes(&format!("/assets/{asset}/utxos?page=2&count=1"))
            .await;

        assert_eq!(status_1, StatusCode::OK);
        assert_eq!(status_2, StatusCode::OK);

        let page_1 = parse_utxos(&bytes_1);
        let page_2 = parse_utxos(&bytes_2);
        assert_eq!(page_1.len(), 1);
        assert_eq!(page_2.len(), 1);

        let key = |x: &AssetUtxoContentInner| format!("{}#{}", x.tx_hash, x.output_index);
        assert_ne!(key(&page_1[0]), key(&page_2[0]));
    }

    #[tokio::test]
    async fn assets_by_subject_utxos_order() {
        let app = TestApp::new();
        let asset = app.vectors().asset_unit.as_str();

        let (status, bytes) = app
            .get_bytes(&format!("/assets/{asset}/utxos?order=asc"))
            .await;
        assert_eq!(status, StatusCode::OK);
        let asc = parse_utxos(&bytes);

        let (status, bytes) = app
            .get_bytes(&format!("/assets/{asset}/utxos?order=desc"))
            .await;
        assert_eq!(status, StatusCode::OK);
        let desc = parse_utxos(&bytes);

        assert!(asc.len() > 1);
        let position = |x: &AssetUtxoContentInner| {
            let (_, tx_index) = app.vectors().tx_position(&x.tx_hash);
            (x.block_height, tx_index, x.output_index)
        };
        let asc_pos: Vec<_> = asc.iter().map(position).collect();
        assert!(asc_pos.windows(2).all(|w| w[0] < w[1]));

        let mut reversed = asc;
        reversed.reverse();
        assert_eq!(desc, reversed);
    }

    #[tokio::test]
    async fn assets_by_subject_utxos_height_constrained() {
        let app = TestApp::new();
        let asset = app.vectors().asset_unit.as_str();

        let (status, bytes) = app.get_bytes(&format!("/assets/{asset}/utxos")).await;
        assert_eq!(status, StatusCode::OK);
        let all = parse_utxos(&bytes);
        let heights: std::collections::BTreeSet<_> = all.iter().map(|x| x.block_height).collect();
        assert!(heights.len() > 1, "need UTxOs in more than one block");
        let pivot = *heights.iter().nth(1).expect("second height");

        // inclusive on both ends
        let (status, bytes) = app
            .get_bytes(&format!("/assets/{asset}/utxos?from={pivot}&to={pivot}"))
            .await;
        assert_eq!(status, StatusCode::OK);
        let within = parse_utxos(&bytes);
        assert!(!within.is_empty());
        assert!(within.iter().all(|x| x.block_height == pivot));

        // one-sided bounds
        let (status, bytes) = app
            .get_bytes(&format!("/assets/{asset}/utxos?from={pivot}"))
            .await;
        assert_eq!(status, StatusCode::OK);
        let from_pivot = parse_utxos(&bytes);
        assert!(from_pivot.iter().all(|x| x.block_height >= pivot));
        assert!(from_pivot.len() < all.len());

        let (status, bytes) = app
            .get_bytes(&format!("/assets/{asset}/utxos?to={pivot}"))
            .await;
        assert_eq!(status, StatusCode::OK);
        let to_pivot = parse_utxos(&bytes);
        assert!(to_pivot.iter().all(|x| x.block_height <= pivot));
        assert_eq!(from_pivot.len() + to_pivot.len(), all.len() + within.len());

        // a range past the tip is an empty page, not an error
        let (status, bytes) = app
            .get_bytes(&format!("/assets/{asset}/utxos?from=99999999"))
            .await;
        assert_eq!(status, StatusCode::OK);
        assert!(parse_utxos(&bytes).is_empty());
    }

    /// A `:index` on a bound cuts inside the edge block by tx index,
    /// inclusive, like on the other endpoints taking `from` / `to`.
    #[tokio::test]
    async fn assets_by_subject_utxos_height_and_index_constrained() {
        let app = TestApp::new();
        let asset = app.vectors().asset_unit.as_str();
        let tx_index = |x: &AssetUtxoContentInner| app.vectors().tx_position(&x.tx_hash).1;

        let (status, bytes) = app.get_bytes(&format!("/assets/{asset}/utxos")).await;
        assert_eq!(status, StatusCode::OK);
        let all = parse_utxos(&bytes);

        // a block holding the asset in more than one tx
        let (height, indexes) = all
            .iter()
            .map(|x| (x.block_height, tx_index(x)))
            .into_group_map()
            .into_iter()
            .map(|(height, indexes)| {
                let indexes: std::collections::BTreeSet<_> = indexes.into_iter().collect();
                (height, indexes)
            })
            .find(|(_, indexes)| indexes.len() > 1)
            .expect("need a block with the asset in more than one tx");
        let cut = *indexes.iter().nth(1).expect("second tx index");
        let in_block: Vec<_> = all.iter().filter(|x| x.block_height == height).collect();

        let (status, bytes) = app
            .get_bytes(&format!(
                "/assets/{asset}/utxos?from={height}:{cut}&to={height}"
            ))
            .await;
        assert_eq!(status, StatusCode::OK);
        let from_cut = parse_utxos(&bytes);
        assert!(!from_cut.is_empty());
        assert!(from_cut
            .iter()
            .all(|x| x.block_height == height && tx_index(x) >= cut));

        let (status, bytes) = app
            .get_bytes(&format!(
                "/assets/{asset}/utxos?from={height}&to={height}:{cut}"
            ))
            .await;
        assert_eq!(status, StatusCode::OK);
        let to_cut = parse_utxos(&bytes);
        assert!(!to_cut.is_empty());
        assert!(to_cut
            .iter()
            .all(|x| x.block_height == height && tx_index(x) <= cut));

        // the two halves overlap exactly on the rows of the cut tx
        let at_cut = in_block.iter().filter(|x| tx_index(x) == cut).count();
        assert_eq!(from_cut.len() + to_cut.len(), in_block.len() + at_cut);
    }

    #[tokio::test]
    async fn assets_by_subject_utxos_bad_request() {
        let app = TestApp::new();
        for asset in invalid_assets() {
            let path = format!("/assets/{asset}/utxos");
            assert_status(&app, &path, StatusCode::BAD_REQUEST).await;
        }

        let asset = app.vectors().asset_unit.as_str();
        for query in ["from=abc", "from=999999999&to=1", "order=a", "count=0"] {
            let path = format!("/assets/{asset}/utxos?{query}");
            assert_status(&app, &path, StatusCode::BAD_REQUEST).await;
        }
    }

    #[tokio::test]
    async fn assets_by_subject_utxos_pruned_history() {
        let full = TestApp::new();
        let asset = full.vectors().asset_unit.clone();

        let (status, bytes) = full.get_bytes(&format!("/assets/{asset}/utxos")).await;
        assert_eq!(status, StatusCode::OK);
        let all = parse_utxos(&bytes);

        let pruned = TestApp::new_pruned();
        let first_retained = pruned.vectors().blocks[2].block_number as i32;
        let pruned_height = pruned.vectors().blocks[0].block_number;

        // rows of pruned blocks are gone, the rest keeps its order
        let (status, bytes) = pruned.get_bytes(&format!("/assets/{asset}/utxos")).await;
        assert_eq!(status, StatusCode::OK);
        let retained = parse_utxos(&bytes);
        let expected: Vec<_> = all
            .iter()
            .filter(|x| x.block_height >= first_retained)
            .cloned()
            .collect();
        assert!(!expected.is_empty(), "fixture needs retained rows");
        assert_eq!(retained, expected);

        // a pruned lower bound is no bound: everything held is newer
        let (status, bytes) = pruned
            .get_bytes(&format!("/assets/{asset}/utxos?from={pruned_height}"))
            .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(parse_utxos(&bytes), expected);

        // a pruned upper bound excludes everything held
        let (status, bytes) = pruned
            .get_bytes(&format!("/assets/{asset}/utxos?to={pruned_height}"))
            .await;
        assert_eq!(status, StatusCode::OK);
        assert!(parse_utxos(&bytes).is_empty());

        // an upper bound past the tip is no bound
        let (status, bytes) = pruned
            .get_bytes(&format!("/assets/{asset}/utxos?to=99999999"))
            .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(parse_utxos(&bytes), expected);

        // pages stay aligned: page 1 of size 1 is the first retained row
        let (status, bytes) = pruned
            .get_bytes(&format!("/assets/{asset}/utxos?count=1&page=1"))
            .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(parse_utxos(&bytes), vec![expected[0].clone()]);
    }

    #[tokio::test]
    async fn assets_by_subject_utxos_not_found() {
        let app = TestApp::new();
        let path = format!("/assets/{}/utxos", missing_asset());
        assert_status(&app, &path, StatusCode::NOT_FOUND).await;
    }

    #[tokio::test]
    async fn assets_by_subject_utxos_internal_error() {
        let app = TestApp::new_with_fault(Some(TestFault::StateStoreError));
        let asset = app.vectors().asset_unit.as_str();
        let path = format!("/assets/{asset}/utxos");
        assert_status(&app, &path, StatusCode::INTERNAL_SERVER_ERROR).await;
    }

    #[tokio::test]
    async fn assets_by_subject_utxos_scan_limit() {
        let app = TestApp::new_with_scan_limit(
            SyntheticBlockConfig {
                block_count: 5,
                txs_per_block: 3,
                ..Default::default()
            },
            3,
        );
        let asset = app.vectors().asset_unit.as_str();

        // a page within the budget is served
        let path = format!("/assets/{asset}/utxos?count=3&page=1");
        assert_status(&app, &path, StatusCode::OK).await;

        // one reaching past it is refused, not cut short
        let path = format!("/assets/{asset}/utxos?count=2&page=2");
        assert_status(&app, &path, StatusCode::BAD_REQUEST).await;
    }
}
