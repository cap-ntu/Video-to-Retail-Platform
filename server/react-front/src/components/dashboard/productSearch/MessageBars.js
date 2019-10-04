import React from 'react';
import { connect } from 'react-redux';
import Snackbar from '../../common/Snackbar';

const mapStateToProps = state => ({
  insertState: state.product.insert,
  searchState: state.product.search,
});

const MessageBars = ({ insertState, searchState }) => (
  <React.Fragment>
    <Snackbar state={searchState} message={{ success: 'Search success.' }} />
    <Snackbar state={insertState} message={{ success: 'Insert success.' }} />
  </React.Fragment>
);

export default connect(mapStateToProps)(MessageBars);
