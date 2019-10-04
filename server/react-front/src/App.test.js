import React from 'react';
import ReactDOM from 'react-dom';
import RootRouter from "./urls";

it('renders without crashing', () => {
  const div = document.createElement('div');
  ReactDOM.render(<RootRouter/>, div);
  ReactDOM.unmountComponentAtNode(div);
});
